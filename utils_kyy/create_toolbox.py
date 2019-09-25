from deap import base, creator
from deap import tools


import random
from itertools import repeat
from collections import Sequence

# For evaluate function --------------------------
import glob
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn    # for hardware tunning (cudnn.benchmark = True)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from thop import profile
from thop import clever_format

import logging

# custom package in utils_kyy
from utils_kyy.utils_graph import load_graph
from utils_kyy.models import RWNN
from utils_kyy.train_validate import train, validate
from utils_kyy.lr_scheduler import LRScheduler
# -------------------------------------------------

# create the toolbox with the right parameters
def create_toolbox_for_NSGA_RWNN(num_graph, args_train, stage_pool_path, log_file_name=None):
    # => Min ( -val_accuracy(top_1),  flops )
    creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0 ))  # name, base (class), attribute // 
    creator.create('Individual', list, fitness=creator.FitnessMin)  # creator.FitnessMaxMin attribute로 가짐    
    
    #####################################
    # Initialize the toolbox
    #####################################
    toolbox = base.Toolbox()
    
    IND_SIZE = 3    # 한 individual, 즉 하나의 chromosome은 3개의 graph. 즉, 3개의 stage를 가짐.

    # toolbox.attribute(0, (num_graph-1)) 이렇게 사용함.
    # 즉, 0 ~ (num_grpah - 1) 중 임의의 정수 선택. => 이걸 3번하면 하나의 small graph가 생김
    BOUND_LOW = 0
    BOUND_UP = num_graph-1
    toolbox.register('attr_int', random.randint, BOUND_LOW, BOUND_UP)   # register(alias, method, argument ...)

    # toolbox.attribute라는 함수를 n번 시행해서 containter인 creator.individual에 넣은 후 해당 instance를 반환함.
    # e.g. [0, 1, 3] 반환
    toolbox.register('individual', tools.initRepeat,
                     creator.Individual, toolbox.attr_int, n=IND_SIZE)

    toolbox.register('population', tools.initRepeat,
                     list, toolbox.individual)    # n은 생략함. toolbox.population 함수를 뒤에서 실행할 때 넣어줌.    
    
    # crossover
    toolbox.register('mate', tools.cxTwoPoint)  # crossover

    # mutation
    toolbox.register('mutate', mutUniformInt_custom, low=BOUND_LOW, up=BOUND_UP)

    # selection
    # => return A list of selected individuals.
    toolbox.register('select', tools.selNSGA2, nd='standard')  # selection.  // k – The number of individuals to select. k는 함수 쓸 때 받아야함    
    
    # evaluate
    toolbox.register('evaluate', evaluate,
                    args_train=args_train, stage_pool_path=stage_pool_path, log_file_name=log_file_name)
    
    return toolbox


############################
# Mutate
############################
# 기존 mutUniformInt에 xrange() 함수가 사용됐어서, range로 수정함.
# indpb: toolbox.mutate() 함수로 사용할 때, MUTPB로 넣어줌. individual의 각 원소에 mutation 적용될 확률.
# indpb – Independent probability for each attribute to be mutated.
def mutUniformInt_custom(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.randint(xl, xu)

    return individual,


############################
# Evaluate
############################
"""
# fitness function
    input: [0, 5, 10]   하나의 크로모좀.

    1) input인 [0, 5, 10]을 받아서 (0번째, 5번째, 10번째)에 해당하는 그래프 파일 각각 읽어와서 신경망 구축
    
    2) training (임시로 1 epoch. 실제 실험 시, RWNN과 같은 epoch 학습시키기)
    
    3) return flops, val_accuracy

"""
def evaluate(individual, args_train, stage_pool_path, channels=109, log_file_name=None):  # individual
    # list 형식의 individual 객체를 input으로 받음   e.g. [0, 4, 17] 
    # 1) load graph
    total_graph_path = glob.glob(stage_pool_path + '*.yaml')    # list
    
    stage_1_graph = load_graph(total_graph_path[individual[0]])
    stage_2_graph = load_graph(total_graph_path[individual[1]])
    stage_3_graph = load_graph(total_graph_path[individual[2]])
    
    graphs = EasyDict({'stage_1': stage_1_graph,
                       'stage_2': stage_2_graph,
                       'stage_3': stage_3_graph
                      })

    # 2) build RWNN
    channels = channels
    NN_model = RWNN(net_type='small', graphs=graphs, channels=channels)
    NN_model.cuda()
    
    ###########################
    # Flops 계산 - [Debug] nn.DataParallele (for multi-gpu) 적용 전에 확인.
    ###########################
    input_flops = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(NN_model, inputs=(input_flops, ), verbose=False)
    
    # 3) Prepare for train
    NN_model = nn.DataParallel(NN_model)  # for multi-GPU
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(NN_model.parameters(), args_train.base_lr,
                                momentum=args_train.momentum,
                                weight_decay=args_train.weight_decay)
    
    start_epoch  = 0
    best_prec1 = 0    
    
    cudnn.benchmark = True    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.  
    
    ###########################
    # Dataset & Dataloader
    ###########################
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),   # 추가함
            transforms.Resize(224),    # 추가함.  imagenet dataset과 size 맞추기
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # rescale 0 ~ 1 => -1 ~ 1
        ])

    val_transform = transforms.Compose(
        [
            transforms.Resize(224),    # 추가함.  imagenet dataset과 size 맞추기
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # rescale 0 ~ 1 => -1 ~ 1
        ])


    # 이미 다운 받아놨으니 download=False
    # 데이터가 없을 경우, 처음엔느 download=True 로 설정해놓고 실행해주어야함
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=train_transform)

    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args_train.batch_size,
                                              shuffle=True, num_workers=args_train.workers)  

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args_train.batch_size,
                                             shuffle=False, num_workers=args_train.workers)    
    
    ###########################
    # Train
    ###########################
    niters = len(train_loader)

    lr_scheduler = LRScheduler(optimizer, niters, args_train)  # (default) args.step = [30, 60, 90], args.decay_factor = 0.1, args.power = 2.0    
    
    for epoch in range(start_epoch, args_train.epochs):
        # train for one epoch
        train(train_loader, NN_model, criterion, optimizer, lr_scheduler, epoch, args_train.print_freq, log_file_name)

        # evaluate on validation set
        prec1 = validate(val_loader, NN_model, criterion, epoch, log_file_name)
        
        # remember best prec@1 and save checkpoint
#         is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
    
    return -best_prec1, flops   # Min (-val_accuracy, flops) 이므로 val_accuracy(top1)에 - 붙여서 return