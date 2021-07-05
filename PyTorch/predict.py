#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

import torch
from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader


def get_args(argv=None, verbose=False):
    """
    This routine parse argv (None means command line) and returns the parsed result
    """

    # a parser to get arguments
    parser = argparse.ArgumentParser(description="LeNet-5 inference")

    # dataset
    parser.add_argument("--dataset", default="MNIST",
                        help="dataset to be used; options are MNIST, FashionMNIST, KMNIST, QMNIST")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="mini-batch size")

    # data location (to read) and model location (to save)
    parser.add_argument("--data_root", default="../data",
                        help="the root directory containing the data")
    parser.add_argument("--trained_model", default="trained_models/MNIST_epoch43_acc0.9913.pkl",
                        help="the file name of trained model for inference")

    args = parser.parse_args(argv)

    # print arguments if verbose is True
    if verbose:
        print('Arguments:')
        print('    dataset:', args.dataset)
        print('    batch_size:', args.batch_size)

        print('    data_root:', args.data_root)
        print('    trained_model:', args.trained_model)

    return args


if __name__ == '__main__':
    # parse the command line to get the arguments
    args = get_args(verbose=True)

    # load trained model
    lenet5 = torch.load(args.trained_model)

    # load test dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((32,32)),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                                   ])
        # all the datasets MNIST, FashionMNIST, KMNIST, QMNIST have images of size 28x28
        # in the original paper, LeNet-5 network takes input images of size 32x32; hence we reshape the images to be of size 32x32
        # the pixel values p have been scaled to be tightly within [0,1], i.e. the grayscale values are divided by 255
        # we we further normalize the pixel values by (p-0.5)/0.5; as a result, the normalized pixel values are in [-1,1]
    method_to_call = getattr(mnist, args.dataset)  # by default, args.dataset is 'MNIST' => method_to_call will be mnist.MNIST
    dataset = method_to_call(root=args.data_root, train=False, transform=transform, download=True)  # train=False for the test set

    test_loader = DataLoader(dataset, batch_size=args.batch_size)

    lenet5.eval()  # ensure the evaluation mode (deactivate dropout or batch normalization, in case of any)
    num_correct = 0
    num_sample = 0
    
    with torch.no_grad():
        # we are not training, and hence do not need gradients here
        for idx, (test_x, test_label) in enumerate(test_loader):
            # udpate num_sample
            sample_count = test_label.shape[0]
            num_sample += sample_count

            # prediction
            predict_y = lenet5(test_x.float())
            # predict_y[i,j] is the predicted probability of sample i being of class j
            # predict_y is of shape (sample_count, num_class), where
            # num_class is 10 (10 classes) for MNIST etc., and
            # sample_count is batch_size (by default 500), except for the last iteration,
            # which may have sample_count less than batch_size

            if idx == 0:
                # first time here, malloc confusion matrix
                num_class = predict_y.shape[1]
                confusion_matrix = np.zeros((num_class, num_class), dtype=np.int32)

            # update test accuracy
            _, predict_ys = torch.max(predict_y, axis=-1)
            # the above line gives the same result as: predict_ys = np.argmax(predict_y, axis=-1)
            num_correct += (predict_ys == test_label).sum().item()
            # the above line gives the same result as num_correct += np.sum((predict_ys == test_label).numpy())

            # update confusion matrix
            for label, prediction in zip(test_label, predict_ys):
                confusion_matrix[label, prediction] += 1

        print('Test accuracy: {:.4f} ({}/{})'.format(num_correct/num_sample, num_correct, num_sample))
        print('Confusion matrix:')
        print(confusion_matrix)
        print('Entry (i,j) is the number of cases that class i is predicted as class j.')

