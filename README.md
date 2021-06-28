# A PyTorch Implementation of LeNet-5

This repo contains a PyTorch implementation of LeNet-5 [(LeCun et al., 1998)][lenet5_paper], with
minor modifications from the original paper, as well as flexible settings for the convenience of experiments.

![LeNet-5 network architecture](images/lenet5.png "Image source: [LeCun et al. (1998)][lenet5_paper]")


## Results

With the default setting,
this implementation achieved the validation and test accuracy
of `MNIST`, `FashionMNIST`, `KMNST`, `QMNIST`
listed in the following table.

| Syntax      | Validation accuracy | Test accuracy |
| ----------- | ----------- | ----------- |
| [MNIST][mnist_dataset]      | 99.13% | 98.94%      |
| [Fashion MNIST][fashion_mnist_dataset]   | 90.94% |  90.36%      |
| [KMNIST][kmnist_dataset]   | 97.98% |  94.16%      |
| [QMNIST][qmnist_dataset]   | 99.07% |  98.88%      |

[mnist_dataset]: https://en.wikipedia.org/wiki/MNIST_database
[fashion_mnist_dataset]: https://github.com/zalandoresearch/fashion-mnist
[kmnist_dataset]: https://github.com/rois-codh/kmnist
[qmnist_dataset]: https://github.com/facebookresearch/qmnist

Note that due to randomized weight initialization and randomized mini-batch
data sampling in training a neural network, you may get slightly different
results.


## Requirements

The code has been tested in the following environment:
- python 3.7.10
- numpy 1.20.3
- torch 1.8.1
- torchvision 0.9.1

It should also work with some other package versions.
The code (`*.py`) is hosted in the subdirectory `PyTorch`, where you can try
it after cloning this repo.


## Training

For a list of training arguments, try `python3 train.py -h`
```
$ python3 train.py -h
usage: train.py [-h] [--dataset DATASET] [--activation ACTIVATION]
                [--dropout_rate DROPOUT_RATE] [--optimizer OPTIMIZER]
                [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                [--num_epoch NUM_EPOCH] [--batch_size BATCH_SIZE] [--lr LR]
                [--num_iter_echo NUM_ITER_ECHO] [--data_root DATA_ROOT]
                [--model_path MODEL_PATH]

LeNet-5 training

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset to be used; options are MNIST, FashionMNIST,
                        KMNIST, QMNIST
  --activation ACTIVATION
                        activation function; options include ReLU, LeakyReLU,
                        PReLU, RReLU, ELU, SELU
  --dropout_rate DROPOUT_RATE
                        dropout rate in the 2 fully-connected layers (fc1 and
                        fc2); 0 means no dropout
  --optimizer OPTIMIZER
                        optimizer for training; options are SGD, RMSprop,
                        Adam, and AdamW, where SGD and RMSprop take the
                        momentum option
  --momentum MOMENTUM   this argument is effective only for SGD and RMSprop; 0
                        means no momentum; recommended 0.9
  --weight_decay WEIGHT_DECAY
                        weight decay rate or L2 regularization coefficient
  --num_epoch NUM_EPOCH
                        number of epochs
  --batch_size BATCH_SIZE
                        mini-batch size
  --lr LR               learning rate (fixed)
  --num_iter_echo NUM_ITER_ECHO
                        training loss is printed for every num_iter_echo mini-
                        batch iterations
  --data_root DATA_ROOT
                        the root directory containing the data
  --model_path MODEL_PATH
                        the directory to host the trained models
```

### Remarks:

- The code `train.py` utilizes `torchvision.datasets.mnist` which can
  can process datasets `MNIST`, `FashionMNIST`, `KMNST`, and `QMNIST`
  in a unified manner.
  - By default `--dataset=MNIST`.
  - In all, the images are of shape 28x28, which are resized to be 32x32,
    the input image size of the original LeNet-5 network.
  - For each dataset, the training data contains 60,000 labeled images.
    We use `torch.utils.data.random_split` to partition it into
    50,000 labeled images for training, and 10,000 labeled images for validation.
  - The default data root directory is `../data/`, specified by `--data_root=../data`.
    For the convenience of experiments, the `MNIST` dataset has been copied
    under `../data/MNIST/` in this repo.
  - To try another dataset for the first time, it will be automatically
    downloaded and saved.
    For example, the option `--dataset=FashionMNIST` will download and save
    the data in `../data/FashionMNIST/` for the first to run it.
- The activation functions in the intermediate layers can be specified by `--activation`.
  - By default, `--activation=SELU', whose [self-normalizing][selu_paper] properties
    magically improve the performance of standard feed-forward neural networks
    (all fully-connected layers), and would have value for convolutional
    neural networks for image classification.
  - The options of the activation include various rectified activations `ReLU`, `LeakyReLU`, `PReLU`, and `RReLU`.
    See the [empirical evaluation][xrelu_paper] along with a short
    description of these activations.
- [Dropout][dropout_paper] can be optionally applied to the two fully connected layers before the last one.
  - The dropout rate can be specified by `--dropout`.
  - By default, `--dropout=0`, which means not to apply dropout.
- The training loss is the standard cross-entropy loss. The default optimization setting is as follows.
  - SGD with momentum of 0.9 (`--optimizer=SGD` and `--momentum=0.9`)
  - weight decay of 5e-4 (`--weight_decay=5e-4`)
  - 50 epochs (`--num_epoch=50`)
  - mini-batch size 256 (`--batch_size=256`)
  - learning rate 0.01 (`--lr=0.01`)
- Other options of the optimizer: [`RMSprop`][rmsprop_ref], [`Adam`][adam_paper], and [`AdamW`][adamw_paper].
- The trained models are saved in the directory specified by `--model_path`.
  - By default, `--model_path=mnist_models`.
  - We use a simple model selection scheme: whenever the validation accuracy
    is improved at the end of an epoch, we save the model in a serialized pickle file (`*.pkl`).
  - The model file name will reflect the dataset name and validation accuracy.
    For example, `./mnist_model/MNIST_epoch43_acc0.9913.pkl`.

As an example, the following command will train a LeNet-5 image classifier for `FashionMNIST` by SGD optimizer
without momentum and with learning rate 0.1.

```
python3 train.py --dataset=FashionMNIST --optimizer=SGD --momentum=0 --lr=0.1
```


## Inference

For a list of prediction arguments, try `python3 predict.py -h`
```
$ python3 predict.py -h
usage: predict.py [-h] [--dataset DATASET] [--batch_size BATCH_SIZE]
                  [--data_root DATA_ROOT] [--trained_model TRAINED_MODEL]

LeNet-5 inference

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset to be used; options are MNIST, FashionMNIST,
                        KMNIST, QMNIST
  --batch_size BATCH_SIZE
                        mini-batch size
  --data_root DATA_ROOT
                        the root directory containing the data
  --trained_model TRAINED_MODEL
                        the file name of trained model for inference
```

### Remarks:

- The code `predict.py` also utilizes `torchvision.datasets.mnist` to
  process the data, and the related comments for `train.py` also apply here,
  except for:
  - For each dataset, the labeled images are from the test set, different
    from those used for training and validation.
  - Each of `MNIST`, `FashionMNIST`, and `KMNIST` has 10,000 labeled images in the test set,
    whereas the test set of `QMNIST` contains 60,000 labeled images.
- The data root directory is specified by `--data_root`.
  - It may be convenient to follow the `data_root` used for training, so
    that the data won't be downloaded again.
  - By default, `--data_root=../data` as that in `train.py`.
- The trained model for inference is specified by `--trained_model`.
  - For example, `--trained_model=./mnist_model/MNIST_epoch43_acc0.9913.pkl`.
  - The following trained models have been provided under `./trained_models/`:
    - `MNIST_epoch43_acc0.9913.pkl`
    - `FashionMNIST_epoch38_acc0.9094.pkl`
    - `KMNIST_epoch43_acc0.9798.pkl`
    - `QMNIST_epoch46_acc0.9907.pkl`
- The prediction is performed in mini-batches, whose size is specified by `--batch_size`.
    - Note that the number of samples in the test set (10,000) should be divisible by `batch_size`.
    - By default, `--batch_size=500`.
    - The classification accuracy of the whole test set will be summarized.

As an example, the following command will calculate and report
the test accuracy of `MNIST` image classification
by the LeNet-5 model `MNIST_epoch43_acc0.9913.pkl`.

```
python3 predict.py --dataset=MNIST --trained_model=trained_models/MNIST_epoch43_acc0.9913.pkl
```


## References

[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)]
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner,
"Gradient-based learning applied to document recognition,"
Proceedings of the IEEE, 86(11):2278-2324, Nov. 1998.

[lenet5_paper]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

[[2](https://papers.nips.cc/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf)]
G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter
"Self-Normalizing Neural Networks,"
Advances in Neural Information Processing Systems, Vol. 30, 2017.

[selu_paper]: https://papers.nips.cc/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf

[[3](https://arxiv.org/pdf/1505.00853.pdf)]
B. Xu, N. Wang, T. Chen, and M. Li,
Empirical Evaluation of Rectified Activations in Convolutional Network,"
arXiv preprint arXiv:1505.00853, 2015.

[xrelu_paper]: https://arxiv.org/pdf/1505.00853.pdf

[[4](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)]
N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov,
"Dropout: A Simple Way to Prevent Neural Networks from Overfitting,"
Journal of Machine Learning Research, 15(56):1929-1958, 2014.

[dropout_paper]: https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

[[5](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)]
T. Tieleman and G. Hinton,
"Lecture 6e - RMSProp,"
COURSERA: Neural Networks for Machine Learning, 2012.

[rmsprop_ref]: https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf

[[6](https://arxiv.org/pdf/1412.6980.pdf)]
D. P. Kingma and J. Ba,
"Adam: A Method for Stochastic Optimization,"
International Conference for Learning Representation, 2015.

[adam_paper]: https://arxiv.org/pdf/1412.6980.pdf

[[7](https://arxiv.org/pdf/1711.05101.pdf)]
I. Loshchilov and F. Hutter,
"Decoupled Weight Decay Regularization,"
International Conference for Learning Representation, 2019.

[adamw_paper]: https://arxiv.org/pdf/1711.05101.pdf
