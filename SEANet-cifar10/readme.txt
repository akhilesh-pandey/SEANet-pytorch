Original code taken from : https://github.com/moskomule/senet.pytorch
 
Usage:
1. Use the main_seanet_cifar10.ipynb ipython-notebook file
 or
2. If you want to use commandline then following the following instructions:
      $ python cifar.py --reduction 8

		reduction=  8 gives the best result. In the paper reduction is denoted by 'r'.
The model's state is saved with name 'best_model.pth' along with the epoch, best_acc1 and optimizer state.

The highest accuracy that we have got is CIFAR-10 is 95.7.
 
The model was trained on GPU - Tesla K-40.

