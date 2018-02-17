import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from skimage import data
import numpy as np
import pandas as pd
import seaborn as sns

import PIL.Image as Image
from PIL.Image import Image as PILImage

from collections import OrderedDict
import time
from pathlib import Path


"""Labels for display in jupyter notebooks."""
SHORT_CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(net, optimizer, loader, epochs, lr, decay, save=None):
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        net.cuda()
        torch.backends.cudnn.benchmark = True
    
    criterion= nn.CrossEntropyLoss()
    
    print_training_info(loader, epochs, optimizer, lr, decay)
    
    refresh = 50
    start_time = time.time()
    net.train()
    losses = [[] for _ in range(epochs)]
    for epoch in range(epochs):
        try:
            acc = []
            for i, (images, labels) in enumerate(loader):
                images, labels = Variable(images), Variable(labels)
                if has_cuda: images, labels = images.cuda(), labels.cuda()
                
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                acc.append(100 * (predicted == labels).cpu().data.sum() / labels.size(0))
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses[epoch].append(loss.cpu().data.numpy().sum())
                if (i+1) % refresh == 0 or (i + 1) == len(loader):
                    np_losses = np.asarray(losses[epoch])
                    a, b = max(0, i - refresh), max(refresh, i)
                    avg_acc = sum(acc[a:b]) / len(acc[a:b])
                    print('Epoch [{}/{}], Iter [{:>4}/{}]  ->  Loss: {:.4f}  (Accuracy: {:.1f} %)'.format(
                        (epoch+1), epochs, (i+1), len(loader), np_losses[a:b].mean(), avg_acc))
        except KeyboardInterrupt:
            if cancel_training(net, save):
                return net, []
            break
    
    all_loss = []
    for epoch_loss in losses:
        all_loss.extend(epoch_loss)
    
    elapsed = time.time() - start_time
    checkpoint = checkpoint_filename(save, epochs, optimizer, lr, decay) if save else None
    finish_training(net, save, checkpoint, elapsed, len(all_loss))
    return net, all_loss

def default_optimizer(lr):
    return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=decay, nesterov=True)

def run_time(elapsed, batches=None, precision=2):
    if batches: elapsed /= batches
    hours = int(elapsed // (3600))
    minutes = int((elapsed - hours * 3600) // 60)
    scale = 10**precision
    seconds = round(scale * (elapsed - hours * 3600 - minutes * 60)) / scale
    return hours, minutes, seconds

def save_state(net, filename, label='save file'):
    if isinstance(filename, str) and filename.endswith('.pkl'):
        torch.save(net.state_dict(), filename)
        print('Created', label, 'at:', filename)
    else:
        print('Failed to create', label, 'at:', filename)

def checkpoint_filename(save, eps, optim, lr, dcy):
    chk_fmt = '{base}-{timestamp:.0f}-e{epochs}-{optim}-lr{lr:.6f}-d{decay:.7f}'
    return chk_fmt.format(
        base=save[:-4],  # i.e. w/o '.pkl'
        timestamp=time.time(),
        epochs=eps,
        optim=optim.__class__.__name__.lower(),
        lr=lr,
        decay=dcy).replace('.', '_') + '.pkl'

def print_training_info(loader, epochs, optim, lr, decay):
    print('Start Time:', time.strftime('%I:%M:%S %p'))
    print('Epochs:', epochs)
    print('Batchsize:', loader.batch_size)
    print('Optimizer:', optim.__class__.__name__)
    print('Learning Rate:', lr)
    print('Weight Decay:', decay)

def finish_training(net, save, checkpoint, training_dur, batches):
    save_state(net, save)
    save_state(net, checkpoint, 'checkpoint')
    h, m, s = run_time(training_dur)
    avg_h, avg_m, avg_s = run_time(training_dur, batches)
    
    print('End Time:', time.strftime("%I:%M:%S %p"))
    print('Run Time:', *((h, 'hours') if h > 0 else ()), m, 'min', s, 'sec')
    avg_h_tuple = (avg_h, 'hours') if avg_h > 0 else ()
    avg_min_tuple = (avg_m, 'min') if avg_m > 0 else ()
    print('Average Batch Time:', *avg_h_tuple, *avg_min_tuple, avg_s, 'sec')

def cancel_training(net, reload):
    if input('Save results of current training (y/[n])? ').lower().startswith('y'): return
    
    print('Training canceled.')
    if Path(reload).exists():
        net.load_state_dict(torch.load(reload))
        print('Reloaded current save.')
    else:
        print('Failed to reload last save:', reload)
    return True

def evaluate(net, loader, save_file=None, stop=None):
    num_batches = 10000 // loader.batch_size
    correct, total = 0, 0
    confusion = np.zeros((10, 10), dtype=int)
    if torch.cuda.is_available(): net.cuda()
    net.eval()
    for i, (images, labels) in enumerate(loader):
        try:
            if stop is not None and i == stop: break
            images = Variable(images, volatile=True)
            if torch.cuda.is_available(): images, labels = images.cuda(), labels.cuda()
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for (y_hat, y) in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                confusion[y_hat, y] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum()
            if (i + 1) % 10 == 0 or total == 10000:
                batch_count = int(total / loader.batch_size)
                print('Accuracy after [', batch_count, '/', num_batches, '] batches: {:.3f}'.format(100 * correct / total), '%')
        except KeyboardInterrupt:
            break
            
    accuracy = int(100 * correct / total)
    print('\nTest Accuracy of the model: {:.2f}'.format(accuracy))
    
    df = pd.DataFrame(data=confusion, index=SHORT_CLASS_NAMES, columns=SHORT_CLASS_NAMES)
    if save_file:
        result_file = save_file[:-4] + '-{}-result-acc{}.csv'.format(int(time.time()), accuracy)
        df.to_csv(result_file)

    colormap = sns.diverging_palette(10, 160, l=75, as_cmap=True)
    return (df.style
            .set_caption('Rows -> Prediction  ::  Columns -> Actual')
            .background_gradient(cmap=colormap))
