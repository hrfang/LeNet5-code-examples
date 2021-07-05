#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

import torch
from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, RMSprop, Adam, AdamW

from lenet5 import LeNet5


def get_args(argv=None, verbose=False):
    """
    This routine parse argv (None means command line) and returns the parsed result
    """

    # a parser to get arguments
    parser = argparse.ArgumentParser(description="LeNet-5 training")

    # device
    parser.add_argument("--device", default="cpu",
                        help="options are cpu, gpu, auto, where auto will use GPU if it is available, and otherwise CPU")

    # dataset
    parser.add_argument("--dataset", default="MNIST",
                        help="dataset to be used; options are MNIST, FashionMNIST, KMNIST, QMNIST")

    # activation and dropout
    parser.add_argument("--activation", default='SELU',
                        help="activation function; options include Tanh, ReLU, LeakyReLU, PReLU, RReLU, ELU, SELU")
    parser.add_argument("--dropout_rate", type=float, default=0.,
                        help="dropout rate in the 2 fully-connected layers (fc1 and fc2); 0 means no dropout")

    # optimizer and related hyperparameters
    parser.add_argument("--optimizer", default="SGD",
                        help="optimizer for training; options are SGD, RMSprop, Adam, and AdamW, where SGD and RMSprop take the momentum option")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="this argument is effective only for SGD and RMSprop; 0 means no momentum; recommended 0.9")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay rate or L2 regularization coefficient")

    parser.add_argument("--num_epoch", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (fixed)")
    parser.add_argument("--num_iter_echo", type=int, default=20,
                        help="training loss is printed for every num_iter_echo mini-batch iterations")

    # data location (to read) and model location (to save)
    parser.add_argument("--data_root", default="../data",
                        help="the root directory containing the data")
    parser.add_argument("--model_path", default="mnist_models",
                        help="the directory to host the trained models")

    args = parser.parse_args(argv)

    # print arguments if verbose is True
    if verbose:
        print('Arguments:')
        print('    device:', args.device)
        print('    dataset:', args.dataset)

        print('    activation:', args.activation)
        print('    dropout_rate:', args.dropout_rate)

        print('    optimizer:', args.optimizer)
        print('    momentum:', args.momentum)
        print('    weight_decay:', args.weight_decay)

        print('    num_epoch:', args.num_epoch)
        print('    batch_size:', args.batch_size)
        print('    lr:', args.lr)
        print('    num_iter_echo:', args.num_iter_echo)

        print('    data_root:', args.data_root)
        print('    model_path:', args.model_path)

    return args


if __name__ == '__main__':
    # parse the command line to get the arguments
    args = get_args(verbose=True)

    if args.device == 'gpu':
        if torch.cuda.is_available():
            use_gpu = True
        else:
            sys.exit('Asked device gpu, which is however not available!')
    elif args.device == 'auto' and torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    if use_gpu:
        device = torch.device('cuda:0')

    # the LeNet-5 network
    activation = getattr(torch.nn.modules.activation, args.activation)  # by default, args.dataset is 'SELU' => activation will be orch.nn.modules.activation.SELU
    lenet5 = LeNet5(activation=activation, dropout_rate=args.dropout_rate)
    if use_gpu:
        lenet5.to(device)

    # load dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((32,32)),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                                   ])
        # all the datasets MNIST, FashionMNIST, KMNIST, QMNIST have images of size 28x28
        # in the original paper, LeNet-5 network takes input images of size 32x32; hence we reshape the images to be of size 32x32
        # the pixel values p have been scaled to be tightly within [0,1], i.e. the grayscale values are divided by 255
        # we we further normalize the pixel values by (p-0.5)/0.5; as a result, the normalized pixel values are in [-1,1]
    method_to_call = getattr(mnist, args.dataset)  # by default, args.dataset is 'MNIST' => method_to_call will be mnist.MNIST
    dataset = method_to_call(root=args.data_root, train=True, transform=transform, download=True)

    # split the dataset (60k samples) into training set (50k samples) and validation set (10k samples)
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        # the routine random_split has the default generator=torch.Generator().manual_seed(42)
        # see https://pytorch.org/docs/stable/data.html
        # the fixed seed 42 means that the train-validation split is the same from one run to another
        # the fixed train-validation split helps compare empirical behaviors with various hyperparameters (e.g. optimizer, learning rate)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        # training data is reshuffled at every epoch that potentially improves the prediction accuracy of the trained model
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # training loss and optimizer
    loss_func = CrossEntropyLoss()
        # the routine CrossEntropyLoss has default argument reduction='mean'
        # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    if args.optimizer == 'SGD':
        optimizer = SGD(lenet5.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = RMSprop(lenet5.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lenet5.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(lenet5.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # if the model path does not exist, make the directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    num_best = 0  # the best number of correct predictions in the previous epochs, initialized 0
    for epoch in range(args.num_epoch):
        # training
        lenet5.train()  # switch to training mode (activate dropout or batch normalization, if any)
        for idx, (train_x, train_label) in enumerate(train_loader):
            # zero out the gradient
            optimizer.zero_grad()

            # prediction and loss
            predict_y = lenet5(train_x.to(device) if use_gpu else train_x)
            loss = loss_func(predict_y, train_label.to(device) if use_gpu else train_label)
            if idx % args.num_iter_echo == 0:
                print('mini-batch #{}:, training cross-entropy loss = {:.6g}'.format(idx, loss.item()))

            # backward propagation
            loss.backward()
            # update the model
            optimizer.step()

        # validation
        lenet5.eval()  # switch to evaluation mode (deactivate dropout or batch normalization, if any)
        num_correct = 0
        sum_loss = 0.
        num_sample = 0
        for idx, (val_x, val_label) in enumerate(val_loader):
            # udpate num_sample
            sample_count = val_label.shape[0]
            num_sample += sample_count

            # prediction
            predict_y = lenet5(val_x.to(device) if use_gpu else val_x).detach()
                # no need of back-propagation here, so "detach"

            # validation loss
            loss = loss_func(predict_y, val_label.to(device) if use_gpu else val_label)
            sum_loss += (loss.cpu() if use_gpu else loss).item()*sample_count
                # the routine CrossEntropyLoss has default argument reduction='mean'
                # so we multiply it by sample_count for the "sum"

            # validation accuracy
            _, predict_ys = torch.max(predict_y, axis=-1)
            is_correct = val_label == (predict_ys.cpu() if use_gpu else predict_ys)
            num_correct += is_correct.sum().item()

        print('Epoch #{}: validation cross-entropy loss = {:.6g}, accuracy: {:.4f} ({}/{})'.format(epoch, sum_loss/num_sample, num_correct/num_sample, num_correct, num_sample))

        # we use a simple model selection scheme: when the validation accuracy is improved, save the model
        if num_best < num_correct:
            model_file_name = os.path.join(args.model_path, args.dataset+'_epoch{}_acc{:.4f}.pkl'.format(epoch, num_correct / num_sample))
            print('Validation accuracy is improved; save the model', model_file_name)
            torch.save(lenet5, model_file_name)
            num_best = num_correct

    # we save the last model anyway
    model_file_name = os.path.join(args.model_path, args.dataset+'_last_acc{:.4f}.pkl'.format(num_correct / num_sample))
    print('Training completed; save the last model', model_file_name)
    torch.save(lenet5, model_file_name)
