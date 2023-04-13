import os
import torch
import torchvision
import torchvision.transforms as transforms
from trainer import CapsNetTrainer
import argparse

DATA_PATH = '/content/data/'

# Collect arguments (if any)
parser = argparse.ArgumentParser()

# MNIST or CIFAR?
parser.add_argument('dataset', nargs='?', type=str, default='MNIST', help="'MNIST' or 'CIFAR' (case insensitive).")
# Batch size
parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size.')
# Epochs
parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs.')
# Learning rate
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate.')
# Number of routing iterations
parser.add_argument('--num_routing', type=int, default=3, help='Number of routing iteration in routing capsules.')
# Exponential learning rate decay
parser.add_argument('--lr_decay', type=float, default=0.96, help='Exponential learning rate decay.')
# Select device "cuda" for GPU or "cpu"
parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=['cuda', 'cpu'], help='Device to use. Choose "cuda" for GPU or "cpu".')
# Use multiple GPUs?
parser.add_argument('--multi_gpu', action='store_true', help='Flag whether to use multiple GPUs.')
# Select GPU device
parser.add_argument('--gpu_device', type=int, default=None, help='ID of a GPU to use when multiple GPUs are available.')
# Data directory
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the MNIST or CIFAR dataset. Alternatively you can set the path as an environmental variable $data.')
args = parser.parse_args()

device = torch.device(args.device)

if args.gpu_device is not None:
    torch.cuda.set_device(args.gpu_device)

if args.multi_gpu:
    args.batch_size *= torch.cuda.device_count()

datasets = {
    'MNIST': torchvision.datasets.MNIST,
    'CIFAR': torchvision.datasets.CIFAR10
}
# dataset defaults
split_train = {'train': True}
split_test = {'train': False}
size = 32
mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

if args.dataset.upper() == 'MNIST':
    args.data_path = os.path.join(args.data_path, 'MNIST')
    size = 28
    classes = list(range(10))
    mean, std = ( ( 0.1307,), ( 0.3081,) )
elif args.dataset.upper() == 'CIFAR':
    args.data_path = os.path.join(args.data_path, 'CIFAR')
    size = 32
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    mean, std = ( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
elif args.dataset.upper() == 'JAMONES':
	args.data_path = os.path.join(args.data_path, 'JAMONES')
	classes = list(range(26))
	size = 32
	split_train = {'split': "train"}
	split_test = {'split': "test"}
else:
    raise ValueError('Dataset must be either MNIST or CIFAR')

transform = transforms.Compose([
    # shift by 2 pixels in either direction with zero padding.
    transforms.RandomCrop(size, padding=2),
    transforms.ToTensor(),
    transforms.Normalize( mean, std )
])

loaders = {}
if args.dataset.upper() not in datasets:
	test_size = 0.2
	dataset = torchvision.datasets.ImageFolder(root=args.data_path, transform=transform)
	num_data = len(dataset)
	num_test = int(test_size * num_data)
	num_train = num_data - num_test
	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
	loaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
	loaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
else:	
	trainset = datasets[args.dataset.upper()](
		root=args.data_path, **split_train, download=True, transform=transform)
	loaders['train'] = torch.utils.data.DataLoader(
		trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

	testset = datasets[args.dataset.upper()](
		root=args.data_path, **split_test, download=True, transform=transform)
	loaders['test'] = torch.utils.data.DataLoader(
		testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

print(8*'#', f'Using {args.dataset.upper()} dataset', 8*'#')

# Run
caps_net = CapsNetTrainer(loaders, args.batch_size, args.learning_rate, args.num_routing, args.lr_decay, device=device, multi_gpu=args.multi_gpu)
caps_net.run(args.epochs, classes=classes)
