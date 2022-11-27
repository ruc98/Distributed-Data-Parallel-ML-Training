import os
import random
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

import model as mdl # Local import

seed = 89395
device = "cpu"
torch.set_num_threads(4)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-ip', dest='master_ip', default="10.10.1.1", type=str, 
                            help='Speficy master ip. Default is 10.10.1.1')
    parser.add_argument('--master-port', dest='master_port', default='4000', type=str,
                            help='Specify master port. Default is 4000.')
    parser.add_argument('--num-nodes', dest='size', type=int, 
                            help='Specify the number of nodes to distribute training over.')
    parser.add_argument('--rank', dest='rank', default=get_rank(), type=int, 
                            help='Specify the rank for this machine. The default takes the number from the computer name.')

    args = parser.parse_args()
    return args.master_ip, args.master_port, args.rank, args.size


def get_rank():
    """ 
    Determine the rank from the computer name.
    """
    return int(os.uname().nodename[4])


def test_distributed_setup():
    """
    Print information about distributed setup.
    """
    print(f'Is initialized: {dist.is_initialized()}')
    print(f'Backend: {dist.get_backend()}')
    print(f'World size: {dist.get_world_size()}')
    print(f'Rank: {dist.get_rank()}\n')


def init_distributed_setup(master_ip, master_port, rank, world_size, backend='gloo'):
    """ 
    Initialize the distributed environment. 
    """
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def create_data_loaders(rank, world_size, batch_size):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    
    training_set = datasets.CIFAR10(root="./../../data", train=True,
                                    download=True, transform=transform_train)
    training_sampler = DistributedSampler(training_set, num_replicas=world_size, 
                                            rank=rank, shuffle=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(training_set,
                                               num_workers=2,
                                               batch_size=batch_size,
                                               sampler=training_sampler,
                                               shuffle=False,
                                               pin_memory=True)

    test_set = datasets.CIFAR10(root="./../../data", train=False,
                                download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)    
    return train_loader, test_loader


def sync_gradients(model, world_size):
    """
    Synchronize gradients among all workers by using the mean gradients.
    """
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size


def train_model(model, rank, world_size, 
                train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    # Collect stats
    running_loss = 0.0
    total_time = 0
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):    
        start_time = time.perf_counter_ns()
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        sync_gradients(model, world_size)
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if batch_idx % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

        if 0 < batch_idx < 40:
            total_time += time.perf_counter_ns() - start_time

        if batch_idx == 39:
            print(f'Total time for 1-39 iteration in ns: {total_time}')
            print(f'Average time for 1-39 iteration in ns: {total_time/39.0}')

    return None

    
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    master_ip, master_port, rank, world_size = parse_arguments()
    init_distributed_setup(master_ip, master_port, rank, world_size)
    test_distributed_setup()

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = int(256/world_size) # batch for one node
    
    training_criterion = torch.nn.CrossEntropyLoss().to(device)
    train_loader, test_loader = create_data_loaders(rank, world_size, batch_size)
   
    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    
    # running training for one epoch
    for epoch in range(1):
        train_loader.sampler.set_epoch(epoch)
        train_model(model, rank, world_size, 
                    train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

