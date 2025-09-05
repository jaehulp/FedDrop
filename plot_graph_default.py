import matplotlib.pyplot as plt
import re
import os
import numpy as np

def draw_graph(namespace):
    plt.xlabel(namespace['xlabel'])
    plt.ylabel(namespace['ylabel'])
    plt.plot(namespace['xaxis'], namespace['yaxis'], label=namespace['label'], linewidth=1, linestyle='-')
    plt.grid(True)
    plt.legend()

class dict_object():

    def __init__(self, mode):
        self.mode = mode
        self.values = {
            'Epochs': [],
            'Acc1': [],
            'Acc5': [],
            'Loss': [],
        }

def extracter(line, dictob):

    if "acc1: (avg)" in line:
        values = re.search(r"acc1: \(avg\) ([\d.]+)\s+acc5: \(avg\) ([\d.]+)\s+loss: \(avg\) ([\d.]+)", line)
        dictob.values['Acc1'].append(float(values.group(1)))
        dictob.values['Acc5'].append(float(values.group(2)))
        dictob.values['Loss'].append(float(values.group(3)))    

def parser(log_path, train_dict, test_dict):

    with open(log_path, 'r') as file:
        log_data = file.readlines()
    last_context = None

    for line in log_data:
        line = line.strip()
        if line.startswith("Train"):
            if line.startswith("Trainable params"):
                continue
            last_context = 'train'
            epoch = int(re.search(r'Train \[(\d+)/\d+\]', line).group(1))
            train_dict.values['Epochs'].append(epoch)
            continue
        elif line.startswith('Test'):
            if last_context != 'train':
                epoch = int(re.search(r'Test \[(\d+)/\d+\]', line).group(1))
            test_dict.values['Epochs'].append(epoch)
            last_context = 'test'
            continue

        if last_context == 'train':
            extracter(line, train_dict)
        elif last_context == 'test':
            extracter(line, test_dict)
save_dir = './graph_img/'

plt.figure()


log_path = './output/cifar10/resnet/dirichlet/client50/mrl_topk16/log.txt'
train_dict = dict_object('train')
test_dict = dict_object('test')
parser(log_path, train_dict, test_dict)

namespace = {
    'xlabel': 'Epochs',
    'xaxis': test_dict.values['Epochs'],
    'ylabel': 'Acc1 accuracy',
    'yaxis': test_dict.values['Acc1'],
    'label': 'topk 16'}

draw_graph(namespace)

log_path = './output/cifar10/resnet/dirichlet/client50/mrl_topk32/log.txt'
train_dict = dict_object('train')
test_dict = dict_object('test')
parser(log_path, train_dict, test_dict)

namespace = {
    'xlabel': 'Epochs',
    'xaxis': test_dict.values['Epochs'],
    'ylabel': 'Acc1 accuracy',
    'yaxis': test_dict.values['Acc1'],
    'label': 'topk 32'}

draw_graph(namespace)

log_path = './output/cifar10/resnet/dirichlet/client50/mrl_topk64/log.txt'
train_dict = dict_object('train')
test_dict = dict_object('test')
parser(log_path, train_dict, test_dict)

namespace = {
    'xlabel': 'Epochs',
    'xaxis': test_dict.values['Epochs'],
    'ylabel': 'Acc1 accuracy',
    'yaxis': test_dict.values['Acc1'],
    'label': 'topk 64'}

draw_graph(namespace)

log_path = './output/cifar10/resnet/dirichlet/client50/mrl_topk128/log.txt'
train_dict = dict_object('train')
test_dict = dict_object('test')
parser(log_path, train_dict, test_dict)

namespace = {
    'xlabel': 'Epochs',
    'xaxis': test_dict.values['Epochs'],
    'ylabel': 'Acc1 accuracy',
    'yaxis': test_dict.values['Acc1'],
    'label': 'topk 128'}

draw_graph(namespace)
log_path = './output/cifar10/resnet/dirichlet/client50/mrl_topk256/log.txt'
train_dict = dict_object('train')
test_dict = dict_object('test')
parser(log_path, train_dict, test_dict)

namespace = {
    'xlabel': 'Epochs',
    'xaxis': test_dict.values['Epochs'],
    'ylabel': 'Acc1 accuracy',
    'yaxis': test_dict.values['Acc1'],
    'label': 'topk 256'}

draw_graph(namespace)

plt.title('Acc1-CIFAR10-ResNet18-dirichlet50')
save_path = os.path.join(save_dir, 'acc1_cifar10_resnet18_dirichlet50_mrl_difftopk.png')
plt.savefig(save_path, dpi=300)
plt.close()
