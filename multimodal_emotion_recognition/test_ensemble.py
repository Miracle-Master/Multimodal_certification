# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch
import time
import logging
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

def val_ensemble(data_loader, model, criterion, opt, dist = None ):
    print("val_ensemble")
    #for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modaltiy:
    #'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise' 
    model.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
        print(i)
        data_time.update(time.time() - end_time)
        inputs_visual = inputs_visual.permute(0,2,1,3,4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0]*inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
        print(input_visual)

        targets = targets.to(opt.device)
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual)
            inputs_audio = Variable(inputs_audio)
            targets = Variable(targets)
        outputs = model(inputs_audio, inputs_visual)
        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs_audio.size(0))
        top5.update(prec5, inputs_audio.size(0))

        losses.update(loss.data, inputs_audio.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))


    return losses.avg.item(), top1.avg.item()
if __name__ == '__main__':
    
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    if opt.certification_method == "MMCert":
        opt.result_path = "C:\\Users\\dongs\\Documents\\multimodal-emotion-recognition-main\\results\\RAVDESS_multimodalcnn_15_checkpoint0.pth"
    pretrained = opt.pretrain_path != 'None'    
    
    #opt.result_path = 'res_'+str(time.time())
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
    opt.arch = '{}'.format(opt.model)  
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
            

    #if opt.dataset == 'RAVDESS':
    #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'

    print(opt)
    torch.manual_seed(opt.manual_seed)
    model, parameters = generate_model(opt)
    print("model_created")
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(opt.device)

    video_transform = transforms.Compose([
        transforms.ToTensor(opt.video_norm_value)])

    test_data = get_test_set(opt, spatial_transform=video_transform) 
    print("dataset_created")
    #load best model
    best_state = torch.load(opt.result_path)
    model.load_state_dict(best_state['state_dict'])
    print("model_loaded")
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    test_loss, test_prec1 = val_ensemble(test_loader, model, criterion, opt
                                    )


    print('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
    test_accuracies.append(test_prec1) 
                
