import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(data_dir,
                           val_split=0.1,
                           random_split=True,
                           batch_size=32,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid multi-process iterators over the dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    @param data_dir: path directory to the dataset. Should contain train/test and optional val directory.
    @param val_split: percentage split of the training set used for validation. If val directory is present, this is of no use.
    @param random_split: whether to randomly split train/validation samples.  If val directory is present, this is of no use.
    @param batch_size: how many samples per batch to load.
    @param the validation set. Should be a float in the range [0, 1].
    @param num_workers: number of subprocesses to use when loading the dataset.
    @param pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.

    @return A tuple containing training/validation sample iterator
    """
    error_msg = "[!] val_split should be in the range [0, 1]."
    assert ((val_split >= 0) and (val_split <= 1)), error_msg

    dataset = datasets.MNIST(
        data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    if random_split:
        # random.shuffle(indices)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    # normalize = transforms.Normalize((0.1307,), (0.3081,))
    # trans = transforms.Compose([
    #     transforms.ToTensor(), normalize,
    # ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True,
        transform=transforms.ToTensor()
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
