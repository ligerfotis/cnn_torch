import torch
from torchvision import datasets
from torchvision import transforms


def load_train_data(dataset, batch_size, num_workers):
    if dataset in ["MNIST", "mnist", "Mnist"]:
        # (Down)loading the MNIST dataset
        train_dataset = datasets.MNIST(
            root="./data/datasets/MNIST/train", train=True,
            transform=transforms.ToTensor(),
            download=True)
    elif dataset in ["CIFAR10", "cifar10", "Cifar10", "CIFAR-10"]:
        # (Down)loading the CIFAR10 dataset
        train_dataset = datasets.CIFAR10(
            root="./data/datasets/CIFAR10/train", train=True,
            transform=transforms.ToTensor(),
            download=True)
    elif dataset in ["CIFAR100", "cifar100", "Cifar100", "CIFAR-100"]:
        # (Down)loading the CIFAR100 dataset
        train_dataset = datasets.CIFAR100(
            root="./data/datasets/CIFAR100/train", train=True,
            transform=transforms.ToTensor(),
            download=True)
    elif dataset in ["ImageNet", "Imagenet", "imagenet"]:
        # (Down)loading the ImageNet dataset
        train_dataset = datasets.ImageNet(
            root="./data/datasets/ImageNet/train", train=True,
            transform=transforms.ToTensor())

    # Creating Dataloaders from the training dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, train_dataset.classes


def load_test_data(dataset, batch_size, num_workers):
    if dataset in ["MNIST", "mnist", "Mnist"]:
        # (Down)loading the MNIST dataset
        test_dataset = datasets.MNIST(
            root="./data/datasets/MNIST/test", train=False,
            transform=transforms.ToTensor(),
            download=True)
    elif dataset in ["CIFAR10", "cifar10", "Cifar10", "CIFAR-10"]:
        # (Down)loading the MNIST dataset
        test_dataset = datasets.CIFAR10(
            root="./data/datasets/CIFAR10/test", train=False,
            transform=transforms.ToTensor(),
            download=True)
    elif dataset in ["CIFAR100", "cifar100", "Cifar100", "CIFAR-100"]:
        # (Down)loading the MNIST dataset
        test_dataset = datasets.CIFAR100(
            root="./data/datasets/CIFAR100/test", train=False,
            transform=transforms.ToTensor(),
            download=True)
    elif dataset in ["ImageNet", "Imagenet", "imagenet"]:
        # (Down)loading the ImageNet dataset
        test_dataset = datasets.ImageNet(
            root="./data/datasets/ImageNet/test", train=False,
            transform=transforms.ToTensor(),
            download=True)

    # Creating Dataloaders from the testing dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                              shuffle=False)
    return test_loader, test_dataset


def get_dataset_classes(dataset):
    if dataset in ["MNIST", "mnist", "Mnist"]:
        return tuple([i for i in range(10)])
    elif dataset in ["CIFAR10", "cifar10", "Cifar10", "CIFAR-10"]:
        return ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif dataset in ["CIFAR100", "cifar100", "Cifar100", "CIFAR-100"]:
        return dataset