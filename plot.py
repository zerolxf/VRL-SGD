
import numpy as np
import argparse
import pandas as pd
from numpy import *
import os
import matplotlib
import argparse
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
opt_names = ["minisgd","localsgd", "vrlsgd"]
opt_label = {"vrlsgd":"VRL-SGD","localsgd":"Local SGD","minisgd":"S-SGD"}
color_set = {"red":"#f4433c", "green":"#0aa858", "blue":"#2d85f0", "yellow":"#ffbc32"}
opt_color = {"minisgd":color_set["green"],"vrlsgd":color_set["red"],"localsgd":color_set["yellow"]}
def get_data(solver_names, path,  eps=1e-14, loss_log=True):
    file_list = os.listdir(path)
    result = dict()
    for file_name in file_list:
        if not("csv" in file_name):
            continue
        for opt in solver_names:
            if opt in file_name:
                record_file = path+file_name
                record = pd.read_csv(record_file, header=None, sep=',')
                result[opt] = record.values
    ans = dict()
    for r in result.items():
        v = r[1]
        if v.shape[0] < 10:
            v = v.T
        ans[r[0]] = v
    return ans


def plot_sgd(path, plot_data, title, pos=2, save="result", re_alpha=0.7):
    sz = len(plot_data)
    plt.figure(dpi=1000)
    plt.rc('font',family='Times New Roman') 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    fontsize = 28
    y = plot_data["vrlsgd"][:,pos]
    len_y = len(y)
    xarange = arange(0,101,50)
    if len_y > 200:
        xarange = arange(0,301,100)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)    
    ax.grid(True, linestyle='-.')
    plt.style.use("seaborn-dark") 
    plt.rcParams.update({'font.size':fontsize})
    plt.rcParams.update({'font.size':fontsize,'font.serif':'Times New Roman'})
    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()
    solver_name = "vrlsgd"
    st = args.st
    y = plot_data[solver_name][st:,pos]
    # print("y",y)
    len_y = len(y)
    x = arange(len_y)
    linewidth = 3
    markersize = 12
    markevery = (len_y+99)//100*3
    ax.plot(x, y,  color=opt_color[solver_name], 
            label=opt_label[solver_name], linestyle="-",linewidth=linewidth, alpha=1,marker = 'o',markevery=2*markevery,markersize=markersize)
    solver_name = "minisgd"
    y = plot_data[solver_name][st:,pos]
    len_y = len(y)
    x = arange(len_y)
    ax.plot(x, y,  color=opt_color[solver_name], 
            label=opt_label[solver_name], linestyle="-",alpha=re_alpha,linewidth=linewidth,marker = '>',markevery=3*markevery,markersize=markersize)
    solver_name = "localsgd"
    y = plot_data[solver_name][st:,pos]
    len_y = len(y)
    x = arange(len_y)
    ax.plot(x, y, color=opt_color[solver_name], 
            label=opt_label[solver_name], linestyle="-", linewidth=linewidth,alpha=re_alpha,marker = 'D',markevery=4*markevery,markersize=markersize)
    font1 = {'family' : 'Times New Roman',
    'size'   : fontsize,
    }
    plt.legend(loc='upper right', prop=font1)
    
    plt.savefig("figure/"+save+'.pdf', dpi=1000, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot figure')
    parser.add_argument('--path', default="./", type=str, help='data path')
    parser.add_argument('--pos', default=2, type=int, help='data pos')
    parser.add_argument('--title', default="Ridge Regression, MSD", type=str, help='title')
    parser.add_argument('--save', default="result", type=str, help='save file name')
    parser.add_argument('--alpha', default=0.7, type=float, help='alpha')
    parser.add_argument('--st', default=0, type=int, help='start ')
    args = parser.parse_args()
    path = args.path
    print("path:{}".format(path))
    test_solver = ["vrlsgd","minisgd", "localsgd"]
    plot_data = get_data(test_solver, path, 1e-5, False)
    print("args.pos",args.pos)
    plot_sgd(path, plot_data, args.title, args.pos, args.save, args.alpha)