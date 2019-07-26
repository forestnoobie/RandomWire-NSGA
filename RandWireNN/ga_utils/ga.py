import os
import time
import torch
import logging
import argparse

from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from utils.graph_reader import read_graph
from dataset.dataloader import create_dataloader, MNIST_dataloader, CIFAR10_dataloader

import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import adabound
import itertools
import traceback

from utils.evaluation import validate
from model.model import RandWire

def ga_train(out_dir, chkpt_path, trainset, valset, writer, logger, hp, hp_str, graphs):
    model = RandWire(hp, graphs).cuda()
    if hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    elif hp.train.optimizer == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(),
                                      lr=hp.train.adabound.initial,
                                      final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=hp.train.sgd.lr,
                                    momentum=hp.train.sgd.momentum,
                                    weight_decay=hp.train.sgd.weight_decay)
    else:
        raise Exception("Optimizer not supported: %s" % hp.train.optimizer)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, hp.train.epoch)

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams are different from checkpoint.")
            logger.warning("Will use new hparams.")
        # hp = load_hparam_str(hp_str)
    else:
        logger.info("Starting new training run")
        logger.info("Writing graph to tensorboardX...")
        writer.write_graph(model, torch.randn(7, hp.model.input_maps, 224, 224).cuda())
        logger.info("Finished.")

    try:
        model.train()
        patients = 0
        prev_acc = 0
        for epoch in itertools.count(init_epoch + 1):
            loader = tqdm.tqdm(trainset, desc='Train data loader')
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                writer.log_training(loss, step)
                loader.set_description('Loss %.02f at step %d' % (loss, step))
                step += 1

            # validation
            val_loss, val_acc = validate(model, valset, writer, epoch)

            if prev_acc < val_acc:
                save_path = os.path.join(out_dir, 'chkpt_%03d.pt' % epoch)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)
                logger.info("Current Validation Accuracy : %10f" % val_acc)

                patients = 0
                prev_acc = val_acc
            else:
                patients += 1

            if patients > 10 or int(prev_acc) == 1:
                return prev_acc
            lr_scheduler.step()

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()

    return prev_acc


# 원래는 yaml 파일, checkpoint path, name of model args로 넘기자
## parser만 잘 바꾸면 될듯
def ga_trainer(args, index_list, f_path, f_name):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('-c', '--config', type=str, required=True,
    #                         help="yaml file for configuration")
    #     parser.add_argument('-p', '--checkpoint_path', type=str, default=None, required=False,
    #                         help="path of checkpoint pt file")
    #     parser.add_argument('-m', '--model', type=str, required=True,
    #                         help="name of the model. used for logging/saving checkpoints")
    #     args = parser.parse_args()

    individual_model_name = args.model + "_{}_{}_{}".format(index_list[0], index_list[1],
                                                            index_list[2])

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())
    ## pytoch 모델 저장하는 위치

    pt_path = os.path.join('.', hp.log.chkpt_dir)
    ## 모델 사전에 정의한 모델 이름으로 저장
    out_dir = os.path.join(pt_path, individual_model_name)
    os.makedirs(out_dir, exist_ok=True)

    log_dir = os.path.join('.', hp.log.log_dir)
    log_dir = os.path.join(log_dir, individual_model_name)
    os.makedirs(log_dir, exist_ok=True)

    if args.checkpoint_path is not None:
        chkpt_path = args.checkpoint_path
    else:
        chkpt_path = None

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                                             '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    if hp.data.train == '' or hp.data.val == '':
        logger.error("hp.data.train, hp.data.val cannot be empty")
        raise Exception("Please specify directories of train data.")

    if hp.model.graph0 == '' or hp.model.graph1 == '' or hp.model.graph2 == '':
        logger.error("hp.model.graph0, graph1, graph2 cannot be empty")
        raise Exception("Please specify random DAG architecture.")

    #     graphs = [
    #         read_graph(hp.model.graph0),
    #         read_graph(hp.model.graph1),
    #         read_graph(hp.model.graph2),
    #     ]

    ## 새로 생성한 파일 위치에서 그래프 읽기
    #print(os.path.join(f_path, args.model + '_' + str(7) +'.txt'))
    graphs = [read_graph(os.path.join(f_path, args.model + '_' + str(idx) +'.txt')) for idx in index_list]

    writer = MyWriter(log_dir)

    dataset = hp.data.type
    switcher = {
        'MNIST': MNIST_dataloader,
        'CIFAR10': CIFAR10_dataloader,
        'ImageNet': create_dataloader,
    }
    assert dataset in switcher.keys(), 'Dataset type currently not supported'
    dl_func = switcher[dataset]
    trainset = dl_func(hp, args, True)
    valset = dl_func(hp, args, False)

    val_acc = ga_train(out_dir, chkpt_path, trainset, valset, writer, logger, hp, hp_str, graphs)

    return val_acc
