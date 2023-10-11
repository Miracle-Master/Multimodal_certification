import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os,shutil,json
import argparse

from statsmodels.stats.proportion import proportion_confint
import glob
import os
import copy
from scipy.special import comb
parser = argparse.ArgumentParser()
parser.add_argument("-ablation_ratio_test", type=float, default=0.00215)#for randomized_ablation
parser.add_argument("-ablation_ratio_test1", type=float, default=0.00322)#for MMCert
parser.add_argument("-ablation_ratio_test2", type=float, default=0.001074)#for MMCert
parser.add_argument("-r1_r2_ratio", type=int, default=1)#for MMCert
parser.add_argument("-alpha", type=float, default=0.001)
parser.add_argument("-c", type=float, help="number of test samples", default=58)
parser.add_argument("-num_ablated_inputs", type=int, default=100)
import copy
def estimate_p_lower(alpha,all_measures,p_lower):
    if p_lower<0:
        return 0
    
    all_measures = all_measures.sort()[0]
    #p_lower = 0.5
    #print(all_measures)
    for i in range(0,len(all_measures)):
        pr = 0
        for j in range(0,i+1):
            pr+=comb(len(all_measures),j, exact=True) *(p_lower**j)*((1-p_lower)**(len(all_measures)-j))
            #pr+=(p_lower**j)*((1-p_lower)**(len(all_measures)-j))
        #print(i,pr)
        if pr > alpha:
            if i!=0:
                #print(all_measures[i-1])
                return all_measures[i-1]
            else:
                return 0

            
def certified_measure(args,all_measures,e1,e2,n1,n2,k1,k2):
    alpha = args.alpha
    delta = 1-((comb(e1,k1, exact=True)*comb(e2,k2, exact=True))/(comb(n1,k1, exact=True)*comb(n2,k2, exact=True)))
    p_lower= 0.5-delta
    return estimate_p_lower(alpha,copy.deepcopy(all_measures),p_lower)
"""
#rewrite this funtion for the baseline method (derandomized smoothing)
def certified_measure_baseline(args,all_measures,e1,e2,n1,n2,k1,k2):
    alpha = args.alpha
    delta = 1-((comb(e1,k1, exact=True)*comb(e2,k2, exact=True))/(comb(n1,k1, exact=True)*comb(n2,k2, exact=True)))
    p_lower= 0.5-delta
    return estimate_p_lower(alpha,copy.deepcopy(all_measures),p_lower)
"""
if __name__ == '__main__':
    args = parser.parse_args()
    print("========MSCert=========")
    n1 = 375*1242
    n2 = 375*1242
    k1 = int(n1*args.ablation_ratio_test1)
    k2 = int(n2*args.ablation_ratio_test2)
    all_outputs = torch.load("output\\"+"MMCert"+"_ablation-ratio-test1="+str(args.ablation_ratio_test1)+"_ablation-ratio-test2="+str(args.ablation_ratio_test2)+"_all_outputs.pth")
    all_globalacc = torch.tensor(all_outputs["all_globalacc"]).t()
    all_pre = torch.tensor(all_outputs["all_pre"]).t()
    all_recall = torch.tensor(all_outputs["all_recall"]).t()
    all_F_score= torch.tensor(all_outputs["all_F_score"]).t()
    all_iou = torch.tensor(all_outputs["all_iou"]).t()
    #print(all_globalacc[0])
    #print(torch.min(all_iou[0]),torch.max(all_iou[0]))
    #{"all_globalacc":all_globalacc, "all_pre":all_pre, "all_recall":all_recall, "all_F_score":all_F_score, "all_iou":all_iou}
    
    rs = []
    all_certified_ious = []
    RANGE = 150
    for r in range(RANGE):
        certified_ious = []
        r1 = r
        r2 = args.r1_r2_ratio*r
        e1 = n1-r1
        e2 = n2-r2
        x = 0
        for i in range(args.c):
            #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
            certified_ious.append(certified_measure(args,all_iou[i],e1,e2,n1,n2,k1,k2))
                #certified_baseline+=1
        #print(certified_ious)
        all_certified_ious.append(np.mean(np.array(certified_ious)))
        rs.append(r)
        print(r,np.mean(np.array(certified_ious)))