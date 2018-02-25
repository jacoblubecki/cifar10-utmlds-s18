import torch
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
import pandas as pd
from skimage import io, transform

from collections import OrderedDict
import os
from pathlib import Path

from utils.image import NORMALIZE


"""Required labels for Kaggle submissions."""
KAGGLE_CLASS_NAMES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

""
TRANSFORMS_KAGGLE = transforms.Compose([transforms.ToPILImage(mode='RGB'), transforms.ToTensor(), NORMALIZE])

class PartialKaggleCIFAR10(torch.utils.data.Dataset):

    def __init__(self, root_dir, start, stop, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.start = start
        self.stop = stop
        self.transform = transform

    def __len__(self):
        return self.stop - self.start  # Lazy...

    def __getitem__(self, idx):
        img_number = self.start + idx + 1
        img_file_name = '{}.png'.format(img_number)
        img_name = Path(self.root_dir) / img_file_name
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return img_number, image

def create_submission(net, partial_datasets, output_dir, file_name, save_intermediates=False, batch_size=256):
    """Takes a net and uses it to label the Kaggle CIFAR-10 images.

    :param net: The pytorch model that will label images.
    :param partial_datasets: An iterator that provides KaggleTestCIFAR objects in order.
    :output_file: The location to save the submission CSV.
    """
    output_labels = OrderedDict()
    output_df = pd.DataFrame([], columns=('id', 'label'))

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        net.cuda()
        
    net.eval()
    for i, kaggle_dataset in enumerate(partial_datasets):
        current_labels = {}
        kaggle_loader = torch.utils.data.DataLoader(kaggle_dataset, batch_size=batch_size)
        
        for j, (indices, images) in enumerate(kaggle_loader):
            images = Variable(images, volatile=True)
            if has_cuda:
                images = images.cuda()
        
            # Add labels to dictionary.
            _, labels = torch.max(net(images), 1)
            for (idx, label) in zip(indices, labels):
                output_labels[idx] = KAGGLE_CLASS_NAMES[label.data[0]]
                current_labels[idx] = KAGGLE_CLASS_NAMES[label.data[0]]
        
            if (j + 1) % 10 == 0:
                print('Completed [', j+1, '/', len(kaggle_loader), '] batches.')
        
        labeled_ids = np.asarray(list(zip(*current_labels.items()))).T
        new_df = pd.DataFrame(data=labeled_ids, columns=('id', 'label'))
        output_df = output_df.append(new_df, ignore_index=True)
        
        if save_intermediates:
            intermediate_name = 'part{}_{}'.format(i+1, file_name)
            intermediate_path = Path(output_dir) / intermediate_name
            new_df.to_csv(intermediate_path.absolute(), index=False)

    output_path = Path(output_dir) / file_name
    output_df.to_csv(output_path.absolute(), index=False)
