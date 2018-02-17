import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms

from models import resnext_bilinear
from utils import kaggle, training
from utils.image import NORMALIZE

# Create net and load checkpoint.
checkpoint = Path('../checkpoints/resnext_bilinear_0_9520.pkl')

net = resnext_bilinear.create_model()

if checkpoint.exists():
	# This ensures that the checkpoint will load.
	# Models saved in GPU states will be converted if necessary.
	to_default = lambda storage, location: storage
	net_state = torch.load(checkpoint, map_location=to_default)

	net.load_state_dict(net_state)
	print('Loaded checkpoint file:', checkpoint)
else:
	print('Path not found:', checkpoint)
	sys.exit(1)


# Setup test data loading.
batch_size = 128
pytorch_data_root = '../data'
transforms_testing = transforms.Compose([transforms.ToTensor(), NORMALIZE])

test_dataset = datasets.CIFAR10(root=pytorch_data_root, train=False, transform=transforms_testing)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#training.evaluate(net, test_loader)

kaggle_data_root = '../_kaggle/test'
demo_images = [kaggle.PartialKaggleCIFAR10(kaggle_data_root, 0, 10, transform=kaggle.TRANSFORMS_KAGGLE)]
kaggle.create_submission(net, demo_images,
	output_dir='../submissions',
	file_name='demo_submission.csv')
