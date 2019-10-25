import torch

import time
import logging


# Train for one epoch
def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, print_freq, log_file_name):

    if log_file_name != None:
        logging.basicConfig(filename=log_file_name, level=logging.INFO)
    
    batch_time = AverageMeter()
    epoch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.update(i, epoch)
        
        target = target.cuda(async=True)
        
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('\t - Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, top5=top5))
            
            niter = epoch * len(train_loader) + i
            
            if log_file_name != None:
                logging.info('\t - Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, top5=top5))

        # ## 임시
        # if i == 2 :
        #     break
        #
def validate(val_loader, model, criterion, epoch, log_file_name):
    
    if log_file_name != None:
        logging.basicConfig(filename=log_file_name, level=logging.INFO)
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))        
                
        # measure elapsed time
        validation_time = time.time() - start

        print('##### Validation_time {validation_time:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} #####'
              .format(validation_time=validation_time, top1=top1, top5=top5))

        if log_file_name != None:
            logging.info('##### Validation_time {validation_time:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} #####'
              .format(validation_time=validation_time, top1=top1, top5=top5))
        
        niter = (epoch + 1)

    return top1.avg


    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.max = max(self.val, val)  # max값 먼저 업데이트 후, val 업데이트
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # torch.topk : input, k, dim=None, largest=True, sorted=True => returns top k element
    # returns values list & indices list
    _, pred = output.topk(maxk, 1, True, True)    
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))   # torch.eq: Computes element-wise equality

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)   # input, dim,
        res.append(correct_k.mul_(100.0 / batch_size))
    return res