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
    parser.add_argument("--trained_model", default="trained_models/MNIST_epoch40_acc0.9900.pkl",
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
    for idx, (test_x, test_label) in enumerate(test_loader):
        # udpate num_sample
        sample_count = test_label.shape[0]
        num_sample += sample_count

        # prediction
        predict_y = lenet5(test_x.float()).detach()  # no need of back-propagation here, so "detach"

        # test accuracy
        predict_ys = np.argmax(predict_y, axis=-1)
        _ = predict_ys == test_label
        num_correct += np.sum(_.numpy(), axis=-1)

    print('Test accuracy: {:.4f} ({}/{})'.format(num_correct/num_sample, num_correct, num_sample))
