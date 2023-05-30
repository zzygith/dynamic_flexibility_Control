from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .mine import MINE_Dataset
from .mineMultipleDen import MINEMD_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""


    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'mine' or dataset_name == 'mine_heater_1d':
        dataset = MINE_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'mine_reactorCooler_2d' or dataset_name == 'mine_reactorCooler_5d':
        dataset = MINEMD_Dataset(root=data_path, normal_class=normal_class)

    return dataset
