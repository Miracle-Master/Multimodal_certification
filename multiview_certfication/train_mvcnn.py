import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset,MultiviewImgDatasetBaseline
from models.MVCNN import MVCNN, SVCNN
import glob
import os
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="resnet50")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-ablation_ratio_train", type=float, default=0.05)
parser.add_argument("-ablation_ratio_test", type=float, default=0.01)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet50")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="modelnet40_images_new/*/train")
parser.add_argument("-val_path", type=str, default="modelnet40_images_new/*/test")
parser.add_argument("-certification_method", type=str, default="MMCert")
parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    certification_method = args.certification_method

    paths = glob.glob('modelnet40_images_new_12x/*/train')
    print(paths)
    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    # STAGE 1
    log_dir = certification_method+'_'+args.name+'_stage_1'
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)

    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    n_models_train = args.num_models*args.num_views

    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train,test_mode=False,num_views=args.num_views,ablation_ratio = args.ablation_ratio_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,ablation_ratio = args.ablation_ratio_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
    trainer.train(5)

    # STAGE 2
    log_dir = certification_method+'_'+args.name+'_stage_2'
    create_folder(log_dir)
    cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    del cnet

    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    if certification_method == "MMCert":
        train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False,test_mode=False, num_models=n_models_train, num_views=args.num_views,ablation_ratio = args.ablation_ratio_train)
    if certification_method == "randomized_ablation":
        train_dataset = MultiviewImgDatasetBaseline(args.train_path, scale_aug=False, rot_aug=False,test_mode=False, num_models=n_models_train, num_views=args.num_views,ablation_ratio = args.ablation_ratio_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer
    if certification_method == "MMCert":
        val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,num_views=args.num_views,ablation_ratio = args.ablation_ratio_test)
    if certification_method == "randomized_ablation":
        val_dataset = MultiviewImgDatasetBaseline(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,num_views=args.num_views,ablation_ratio = args.ablation_ratio_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=args.num_views)
    trainer.train(5)


