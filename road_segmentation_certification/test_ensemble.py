import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.util import confusion_matrix, getScores, tensor2labelim, tensor2im, print_current_losses
import numpy as np
import random
import torch
import cv2
from tensorboardX import SummaryWriter
from torch.utils.data import random_split
from options.test_options import TestOptions
import os
from PIL import Image
if __name__ == '__main__':
    #opt = TrainOptions().parse()
    valid_opt = TestOptions().parse()
    #save_dir = os.path.join('output', opt.certification_method+ opt.ablation_ratio_test1 +opt.ablation_ratio_test2)
    
    #if not os.path.exists(save_dir):
        #os.makedirs(save_dir)
    np.random.seed(valid_opt.seed)
    random.seed(valid_opt.seed)
    torch.manual_seed(valid_opt.seed)
    torch.cuda.manual_seed(valid_opt.seed)
    
    #train_data_loader = CreateDataLoader(train_opt)
    all_dataset = create_dataset(valid_opt)#train_data_loader.load_data()
    all_dataset_size = len(all_dataset)
    print('#all images = %d' % all_dataset_size)
    train_length = int(0.8 * all_dataset_size)  # 80% for training
    val_length = all_dataset_size - train_length  # 20% for validation
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(all_dataset, [train_length, val_length])
    print(len(train_dataset),len(val_dataset))
    val_dataset.phase = "test"
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=valid_opt.batch_size,
    shuffle=not valid_opt.serial_batches,
    num_workers=0,
    worker_init_fn=lambda worker_id: numpy.random.seed(valid_opt.seed + worker_id))
    
    valid_opt.isTrain = False
    valid_opt = TestOptions().parse()
    valid_opt.phase = 'test'
    valid_opt.serial_batches = True
    valid_opt.isTrain = False
    val_dataset_size = len(val_dataset)
    valid_opt.batch_size = 1
    print('#validation images = %d' % val_dataset_size)
    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=not valid_opt.serial_batches,
    num_workers=0,
    worker_init_fn=lambda worker_id: numpy.random.seed(opt.seed + worker_id))

    writer = SummaryWriter()

    model = create_model(valid_opt, train_dataset.dataset)
    model.load_networks(valid_opt)
    total_steps = 0
    tfcount = 0
    F_score_max = 0
 
    ### Evaluation on the validation set ###
    model.eval()
    valid_loss_iter = []
    epoch_iter = 0
    
    print("num_labels:", val_dataset.dataset.num_labels)
    all_globalacc =[]
    all_pre =[]
    all_recall =[]
    all_F_score = []
    all_iou = []
    for i in range(100):
        print(i)
        globalaccs =[]
        pres =[]
        recalls =[]
        F_scores = []
        ious = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                conf_mat = np.zeros((val_dataset.dataset.num_labels, val_dataset.dataset.num_labels), dtype=np.float)
                model.set_input(data)
                model.forward()
                model.get_loss()
                epoch_iter += valid_opt.batch_size
                gt = model.label.cpu().int().numpy()
                _, pred = torch.max(model.output.data.cpu(), 1)
                pred = pred.float().detach().int().numpy()

                # Resize images to the original size for evaluation
                image_size = model.get_image_oriSize()
                oriSize = (image_size[0].item(), image_size[1].item())
                gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
                pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
                #print(np.max(gt[0]),np.min(gt[0]))
                #Image.fromarray(gt[0]*100).show()
                #Image.fromarray(pred[0]*100).show()
                #print(gt.shape, pred.shape)
                conf_mat = confusion_matrix(gt, pred, val_dataset.dataset.num_labels)
                globalacc, pre, recall, F_score, iou = getScores(conf_mat)
                globalaccs.append(globalacc)
                pres.append(pre)
                recalls.append(recall)
                F_scores.append(F_score)
                ious.append(iou)
                #print('valid epoch {0:}, iters: {1:}/{2:} '.format(epoch, epoch_iter, len(val_dataset) * valid_opt.batch_size), end='\r')

        print('valid/global_acc', globalaccs[0:5])
        print('valid/pre', pres[0:5])
        print('valid/recall', recalls[0:5])
        print('valid/F_score', F_scores[0:5])
        print('valid/iou', ious[0:5])
        all_globalacc.append(globalaccs)
        all_pre.append(pres)
        all_recall.append(recalls)
        all_F_score.append(F_scores)
        all_iou.append(ious)
    # Save the best model according to the F-score, and record corresponding epoch number in tensorboard
    #if F_score > F_score_max:
    #print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))

    #all_global_acc = torch.stack(all_globalacc)
    #all_pre = torch.stack(all_pre)
    #all_recall = torch.stack(all_recall)
    #all_F_score = torch.stack(all_F_score)
    #all_iou = torch.stack(all_iou)
    dict_ = {"all_globalacc":all_globalacc, "all_pre":all_pre, "all_recall":all_recall, "all_F_score":all_F_score, "all_iou":all_iou}
    if valid_opt.certification_method == "randomized_ablation":
        torch.save(dict_, 'output/'+valid_opt.certification_method+"_ablation-ratio-test="+str(valid_opt.ablation_ratio_test)+'_all_outputs.pth')
    else:
        torch.save(dict_, 'output/'+valid_opt.certification_method+"_ablation-ratio-test1="+str(valid_opt.ablation_ratio_test1)+"_ablation-ratio-test2="+str(valid_opt.ablation_ratio_test2)+'_all_outputs.pth')