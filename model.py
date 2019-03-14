import argparse
import json
import logging
import numpy
import os
import pandas
import sagemaker_containers
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

criterion = nn.TripletMarginLoss()


class Net(nn.Module):
	def __init__(self, embedding_net):
		super(Net, self).__init__()
		self.embedding_net = embedding_net

	def forward(self, anchor, positive, negative):
		a = self.embedding_net(anchor)
		p = self.embedding_net(positive)
		n = self.embedding_net(negative)
		return a, p, n


class EmbeddingNet(nn.Module):
	def __init__(self):
		super(EmbeddingNet, self).__init__()
		self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
		self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
		self.conv3 = nn.Conv1d(128, 256, kernel_size=5)
		self.conv4 = nn.Conv1d(256, 512, kernel_size=5)
		self.conv5 = nn.Conv1d(512, 512, kernel_size=7)
		self.max = nn.MaxPool1d(2)
		self.fc1 = nn.Linear(182 * 512, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, 20)

	def forward(self, x):
		x = F.relu(self.max(self.conv1(x)))
		x = F.relu(self.max(self.conv2(x)))
		x = F.relu(self.max(self.conv3(x)))
		x = F.relu(self.max(self.conv4(x)))
		x = F.relu(self.max(self.conv5(x)))
		x = x.view(-1, 182 * 512)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.dropout(x, training=self.training)
		return self.fc3(x)


class PegaAlertDataset(Dataset):
	def __init__(self, root_dir, csv_file, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.examples = pandas.read_csv(os.path.join(self.root_dir, csv_file))

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		sample = os.path.join(self.root_dir,
							  self.examples.iloc[idx, 2],
							  self.examples.iloc[idx, 0])
		positive = os.path.join(self.root_dir,
								self.examples.iloc[idx, 2],
								self.examples.iloc[idx, 1])
		negative = os.path.join(self.root_dir,
								self.examples.iloc[idx, 4],
								self.examples.iloc[idx, 3])

		with open(sample, "r") as file:
			sample = file.read()
		with open(positive, "r") as file:
			positive = file.read()
		with open(negative, "r") as file:
			negative = file.read()

		if self.transform:
			sample = self.transform(sample)
			positive = self.transform(positive)
			negative = self.transform(negative)

		return sample, positive, negative


class Text2Tensor:
	def __init__(self, max_size=6000):
		self.max_size = max_size

	def __call__(self, text):
		tmp = numpy.fromiter(map(lambda x: ord(x), list(text)), dtype=numpy.double)
		if tmp.size < self.max_size:
			tmp = numpy.append(tmp, numpy.zeros(self.max_size - tmp.size))
		elif tmp.size > self.max_size:
			tmp = tmp[:self.max_size]
		return torch.tensor([tmp])


def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
	logger.info("Get train data loader")
	dataset = PegaAlertDataset(training_dir, 'train.csv', transform=transforms.Compose([
		Text2Tensor()
	]))
	train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
									   sampler=train_sampler, **kwargs)


def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
	logger.info("Get test data loader")
	return torch.utils.data.DataLoader(
		PegaAlertDataset(training_dir, 'test.csv', transform=transforms.Compose([
			Text2Tensor()
		])),
		batch_size=test_batch_size, shuffle=True, **kwargs)


def _average_gradients(model):
	# Gradient averaging.
	size = float(dist.get_world_size())
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
		param.grad.data /= size


def train(args):
	is_distributed = len(args.hosts) > 1 and args.backend is not None
	logger.debug("Distributed training - {}".format(is_distributed))
	use_cuda = args.num_gpus > 0
	logger.debug("Number of gpus available - {}".format(args.num_gpus))
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	device = torch.device("cuda" if use_cuda else "cpu")

	if is_distributed:
		# Initialize the distributed environment.
		world_size = len(args.hosts)
		os.environ['WORLD_SIZE'] = str(world_size)
		host_rank = args.hosts.index(args.current_host)
		os.environ['RANK'] = str(host_rank)
		dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
		logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
			args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
			dist.get_rank(), args.num_gpus))

	# set the seed for generating random numbers
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed(args.seed)

	train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
	test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

	logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
		len(train_loader.sampler), len(train_loader.dataset),
		100. * len(train_loader.sampler) / len(train_loader.dataset)
	))

	logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
		len(test_loader.sampler), len(test_loader.dataset),
		100. * len(test_loader.sampler) / len(test_loader.dataset)
	))

	model = Net(EmbeddingNet()).double()
	model = model.to(device)
	if is_distributed and use_cuda:
		# multi-machine multi-gpu case
		model = torch.nn.parallel.DistributedDataParallel(model)
	else:
		# single-machine multi-gpu case or single-machine or multi-machine cpu case
		model = torch.nn.DataParallel(model)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

	for epoch in range(1, args.epochs + 1):
		model.train()
		for batch_idx, data in enumerate(train_loader, 1):
			data = map(lambda d: d.to(device), data)
			optimizer.zero_grad()
			anchor, positive, negative = model(*data)
			loss = criterion(anchor, positive, negative)
			loss.backward()
			if is_distributed and not use_cuda:
				# average gradients manually for multi-machine cpu case only
				_average_gradients(model)
			optimizer.step()
			if batch_idx % args.log_interval == 0:
				logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
					epoch, batch_idx * len(anchor), len(train_loader.sampler),
					100. * batch_idx / len(train_loader), loss.item()))
		test(model, test_loader, device)
	save_model(model, args.model_dir)


def test(model, test_loader, device):
	model.eval()
	test_loss = 0
	correct = 0
	dist = nn.PairwiseDistance()

	with torch.no_grad():
		for data in test_loader:
			data = map(lambda d: d.to(device), data)
			anchor, positive, negative = model(*data)
			test_loss += criterion(anchor, positive, negative).item()  # sum up batch loss
			correct += (dist(anchor, positive) < dist(anchor, negative)).sum().item()

	test_loss /= len(test_loader.dataset)
	logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


def model_fn(model_dir):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = torch.nn.DataParallel(Net())
	with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
		model.load_state_dict(torch.load(f))
	return model.to(device)


def save_model(model, model_dir):
	logger.info("Saving the model.")
	path = os.path.join(model_dir, 'model.pth')
	# recommended way from http://pytorch.org/docs/master/notes/serialization.html
	torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Data and model checkpoints directories
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--backend', type=str, default=None,
						help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

	# Container environment
	parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
	parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
	parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
	parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
	parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

	train(parser.parse_args())
