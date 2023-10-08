import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from torch.autograd import Variable
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset,MultiviewImgDatasetBaseline, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
import glob
import os
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=15)
parser.add_argument("-ablation_ratio_test", type=float, default=0.05)
parser.add_argument("-ablation_ratio_test1", type=float, default=0.01)
parser.add_argument("-ablation_ratio_test2", type=float, default=0.01)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
parser.add_argument("-num_views", type=int, help="number of views", default=2)
parser.add_argument("-train_path", type=str, default="modelnet40_images_new/*/train")
parser.add_argument("-val_path", type=str, default="modelnet40_images_new/*/test")
parser.add_argument("-model_path", type=str, default=".\mvcnn_stage_2\mvcnn\model-00001.pth")
parser.add_argument("-certification_method", type=str, default="randomized_ablation")
parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
def predict(dataloader,model):
    all_correct_points = 0
    all_points = 0
    model.cuda()
    # in_data = None
    # out_data = None
    # target = None

    wrong_class = np.zeros(40)
    samples_class = np.zeros(40)
    all_loss = 0

    model.eval()

    avgpool = nn.AvgPool1d(1, 1)

    total_time = 0.0
    total_print_time = 0.0
    all_target = []
    all_pred = []
    
    for _, data in enumerate(dataloader, 0):

        N,V,C,H,W = data[1].size()
        in_data = Variable(data[1]).view(-1,C,H,W).cuda()

        target = Variable(data[0]).cuda()

        out_data = model(in_data)
        pred = torch.max(out_data, 1)[1]
        results = pred == target

        for i in range(results.size()[0]):
            if not bool(results[i].cpu().data.numpy()):
                wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
            samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
        correct_points = torch.sum(results.long())

        all_correct_points += correct_points
        all_points += results.size()[0]
        all_pred.extend(pred.cpu())
        all_target.extend(target.cpu())
    print ('Total # of test models: ', all_points)
    acc = all_correct_points.float() / all_points
    val_overall_acc = acc.cpu().data.numpy()

    print ('val overall acc. : ', val_overall_acc)

    return val_overall_acc, torch.tensor(all_pred),torch.tensor(all_target)


if __name__ == '__main__':
    args = parser.parse_args()
    backbone = SVCNN(args.name, nclasses=40, cnn_name=args.cnn_name)
    model = MVCNN(args.name, backbone, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    if args.certification_method == "MMCert":
        model.load_state_dict(torch.load(".\MMCert_mvcnn_stage_2\mvcnn\model-00004.pth"))
        #model.load_state_dict(torch.load("randomized_ablation_mvcnn_stage_2\mvcnn\model-00004.pth"))
    else:
        #model.load_state_dict(torch.load(".\mvcnn_stage_2\mvcnn\model-00001.pth"))
        model.load_state_dict(torch.load("randomized_ablation_mvcnn_stage_2\mvcnn\model-00004.pth"))
    n_models_train = args.num_models*args.num_views
    if args.certification_method == "MMCert":
        val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,num_views=args.num_views,num_models = n_models_train, ablation_ratio1 = args.ablation_ratio_test1,ablation_ratio2 = args.ablation_ratio_test2)
    else: 
        val_dataset = MultiviewImgDatasetBaseline(args.val_path, scale_aug=False, rot_aug=False, test_mode=True,num_views=2,num_models = n_models_train, ablation_ratio = args.ablation_ratio_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    all_preds = []
    all_targets = []
    for i in range(100):
        print(i)
        _, preds,targets = predict(val_loader,model)
        all_preds.append(preds)
        all_targets.append(targets)
    all_preds = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    print(all_preds)
    if args.certification_method == "randomized_ablation":
        torch.save((all_preds,all_targets), 'output/'+args.certification_method+"_ablation-ratio-test="+str(args.ablation_ratio_test)+'_all_outputs.pth')
    else:
        torch.save((all_preds,all_targets), 'output/'+"_ablation-ratio-test1="+str(args.ablation_ratio_test1)+"_ablation-ratio-test2="+str(args.ablation_ratio_test2)+'_all_outputs.pth')