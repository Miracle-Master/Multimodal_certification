import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from torch.autograd import Variable
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN
from statsmodels.stats.proportion import proportion_confint
import glob
import os
import copy
from scipy.special import comb
parser = argparse.ArgumentParser()
parser.add_argument("-num_models", type=int, help="number of models per class", default=15)
parser.add_argument("-ablation_ratio_test", type=float, default=0.01)#for randomized_ablation
parser.add_argument("-ablation_ratio_test1", type=float, default=0.015)#for MMCert
parser.add_argument("-ablation_ratio_test2", type=float, default=0.005)#for MMCert
parser.add_argument("-r1_r2_ratio", type=int, default=6)#for MMCert
parser.add_argument("-num_views", type=int, help="number of views", default=2)
parser.add_argument("-alpha", type=float, default=0.001)
parser.add_argument("-c", type=float, help="number of test samples", default=500)
parser.add_argument("-num_ablated_inputs", type=int, default=100)
parser.add_argument("-num_classes", type=int, default=40)
parser.add_argument("-rho", type=str, default="match_num_pixels")
#parser.add_argument("-output_path", type=str, default=".\mvcnn_stage_2\mvcnn\model-00001.pth")

def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha, method="beta")
def get_bounds(args,counts):
    lower_bounds = np.zeros((args.c,args.num_classes))
    upper_bounds = np.zeros((args.c,args.num_classes))
    for i in range(args.c):
        for j in range(args.num_classes):
            lower_bounds[i][j],upper_bounds[i][j] = _lower_confidence_bound(counts[i][j], args.num_ablated_inputs, args.alpha/args.c)
    return lower_bounds, upper_bounds

def is_certified(true_label,lower_bounds,upper_bounds,e1,e2,n1,n2,k1,k2):
    #print(comb(5, 2, exact=True))
    lower_bounds = copy.deepcopy(lower_bounds)
    upper_bounds = copy.deepcopy(upper_bounds)
    delta = 1-((comb(e1,k1, exact=True)*comb(e2,k2, exact=True))/(comb(n1,k1, exact=True)*comb(n2,k2, exact=True)))
    #delta = 1-((e1/n1)**k1)*((e2/n2)**k2)
    #print(n1-e1,delta)
    lower_bounds -= delta
    upper_bounds += delta
    upper_bounds[true_label] = 0
    #print(n1-e1,delta, lower_bounds[true_label],np.max(upper_bounds))
    if lower_bounds[true_label]>np.max(upper_bounds):
        
        return True
    return False
def is_certified_baseline(true_label,lower_bounds,upper_bounds,e1,e2,n1,n2,k1,k2):
    #print(comb(5, 2, exact=True))
    lower_bounds = copy.deepcopy(lower_bounds)
    upper_bounds = copy.deepcopy(upper_bounds)
    delta = 1-((comb(e1+e2,k1+k2, exact=True))/(comb(n1+n2,k1+k2, exact=True)))
    #delta = 1-((e1+e2)/(n1+n2))**(k1+k2)
    #print(n1-e1,delta)
    lower_bounds -= delta
    upper_bounds += delta
    upper_bounds[true_label] = 0
    #print(n1-e1,delta, lower_bounds[true_label],np.max(upper_bounds))
    if lower_bounds[true_label]>0.5:
        return True
    return False
if __name__ == '__main__':
    args = parser.parse_args()
    print("========MSCert=========")
    if args.r1_r2_ratio == 1:
        all_outputs = torch.load('output/'+"_ablation-ratio-test1="+str(0.01)+"_ablation-ratio-test2="+str(0.01)+'_all_outputs.pth')
    else:
        all_outputs = torch.load('output/'+"_ablation-ratio-test1="+str(args.ablation_ratio_test1)+"_ablation-ratio-test2="+str(args.ablation_ratio_test2)+'_all_outputs.pth')
    all_preds = all_outputs[0].t()
    all_targets=all_outputs[1].t()
    print(all_preds.shape,all_targets.shape)
    all_preds = all_preds[0:args.c]
    all_targets = all_targets[0:args.c]
    n1 = 224*224
    n2 = 224*224
    k1 = int(n1*args.ablation_ratio_test1)
    k2 = int(n2*args.ablation_ratio_test2)

    counts = np.zeros((all_targets.shape[0],args.num_classes))
    for i in range(all_targets.shape[0]):
        for j in range(all_targets.shape[1]):
            counts[i][all_preds[i][j]] += 1
    lower,upper = get_bounds(args, counts)
    #lower = lower-(1/(n**m))
    #upper = upper+(1/(n**m))
    """
    for i in range(10):
        print(counts[i])
        print(lower[i])
        print(upper[i])
    """
    rs = []
    CAs = []
    CAs_baseline = []
    RANGE=30
    for r in range(RANGE):
        r1 = r
        r2 = args.r1_r2_ratio*r
        e1 = n1-r1
        e2 = n2-r2
        x = 0
        certified = 0
        for i in range(args.c):
            #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
            if is_certified(all_targets[i][0],lower[i],upper[i],e1,e2,n1,n2,k1,k2) == True:
                certified+=1

        CAs.append(certified/args.c)
        rs.append(r)
        print(r,certified/args.c)
    print("========Randomized_ablation=========")
    #randomized_ablation  
    all_outputs = torch.load('output/'+"randomized_ablation_ablation-ratio-test="+str(args.ablation_ratio_test)+'_all_outputs.pth')
    all_preds = all_outputs[0].t()
    all_targets=all_outputs[1].t()
    #print(all_preds.shape,all_targets.shape)
    all_preds = all_preds[0:args.c]
    all_targets = all_targets[0:args.c]
    n1 = 224*224
    n2 = 224*224
    k1 = int(n1*args.ablation_ratio_test)
    k2 = int(n2*args.ablation_ratio_test)

    counts = np.zeros((all_targets.shape[0],args.num_classes))
    for i in range(all_targets.shape[0]):
        for j in range(all_targets.shape[1]):
            counts[i][all_preds[i][j]] += 1
    lower,upper = get_bounds(args, counts)
    #lower = lower-(1/(n**m))
    #upper = upper+(1/(n**m))
    """
    for i in range(10):
        print(counts[i])
        print(lower[i])
        print(upper[i])
    """
    rs = []
    CAs_baseline = []
    for r in range(RANGE):
        r1 = r
        r2 = args.r1_r2_ratio*r
        e1 = n1-r1
        e2 = n2-r2
        x = 0

        certified_baseline = 0

        for i in range(args.c):
            #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
            if is_certified_baseline(all_targets[i][0],lower[i],upper[i],e1,e2,n1,n2,k1,k2) == True:
                certified_baseline+=1
        CAs_baseline.append(certified_baseline/args.c)
        rs.append(r)
        print(r,certified_baseline/args.c)
    plt.figure(figsize = (7,4))
    
    plt.plot( rs,CAs, label =  r'MMCert',color ="red")
    #plt.plot( Ts,bagging_recalls, label = 'Bagging Recall',linestyle = '--',color ="b")
    plt.plot( rs,CAs_baseline, label =  r'RA',linestyle = '--',color ="blue")

    #plt.plot( Ts,dpa_ind_recalls, label = 'DPA Recall',linestyle = 'dotted',color ="b")
    #plt.plot( Ts,dpa_ind_recalls, label = 'DPA',linestyle = 'dotted',color ="g")
    #plt.plot( Ts,dpa_ind_precisions, label = 'DPA Precisions',linestyle = '-.',color ="r")
    plt.xlabel(r'$r_1$', fontsize=22)
    plt.ylabel('Certified accuracy', fontsize=22)
    plt.grid()
    plt.legend(loc='upper right',fontsize=20)
    plt.xlim(0)
    plt.xlim(xmax =6)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
    #plt.xticks([0,10,20,30,40,50,60])
    plt.xticks([0,5,10,15,20,25])
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.tight_layout()
    plt.rcParams['figure.dpi'] = 1000
    plt.tight_layout()

    plt.savefig('figs/'+args.rho+'_r1-r2-ratio='+str(args.r1_r2_ratio)+'.pdf', dpi=1000,bbox_inches='tight')
    plt.rcParams['figure.dpi'] = 1000
    plt.show()