from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
np.random.seed(24)
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MaterialFEADataset(Dataset):
    """Material FEA Dataset"""

    def __init__(self, data_file, max_context_num):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.loadtxt(data_file, dtype='float32',delimiter=',')
        self.max_context_num = max_context_num
        
        MAX = np.max(self.data, axis=0)
        MIN = np.min(self.data, axis=0)
        self.norm = ((self.data - MIN) / (MAX - MIN) + 0.1) * 4 - 2.4
        np.random.shuffle(self.norm)
        
        self.data_train = self.norm[:max_context_num]
        self.data_test = self.norm[max_context_num:]

    def train_data(self, diff):
        
        num_context = np.random.randint(self.max_context_num-diff,self.max_context_num) # extract random number of contexts
#        num_target = np.random.randint(0, self.max_context_num - num_context)
#        num_context = self.max_context_num
        num_target = self.max_context_num - num_context
        num_total_points = num_context + num_target # this num should be # of target points
        total_id = np.random.choice(range(self.max_context_num), num_total_points, replace=False)
        context_id = total_id[:num_context]
        
        context_x = [[]]
        context_y = [[]]
        target_x = [[]]
        target_y = [[]]
        
        for i, id in enumerate(total_id):
            target_x[0].append(self.data_train[id,:-1])
            target_y[0].append(self.data_train[id,-1:])
            
        for i, id in enumerate(context_id):
            context_x[0].append(self.data_train[id,:-1])
            context_y[0].append(self.data_train[id,-1:])
            
        return np.array(context_x), np.array(context_y), np.array(target_x), np.array(target_y)
        
    def test_data(self):
        
        num_context = self.max_context_num
        num_target = 20
        num_total_points = num_context + num_target # this num should be # of target points
        
        context_x = [[]]
        context_y = [[]]
        target_x = [[]]
        target_y = [[]]
        
        target_x[0] = np.concatenate((self.norm[:num_context,:-1], self.norm[-num_target:,:-1]), axis = 0)
        target_y[0] = np.concatenate((self.norm[:num_context,-1:], self.norm[-num_target:,-1:]), axis = 0)
            
        context_x[0] = self.data_train[:,:-1]
        context_y[0] = self.data_train[:,-1:]
        
        
        return np.array(context_x), np.array(context_y), np.array(target_x), np.array(target_y)
    
class SpiralDataset(Dataset):
    """Spiral Dataset, with scalar input and vector output"""

    def __init__(self, data_file, max_context_num):
        
        self.data = np.loadtxt(data_file, dtype='float32',delimiter=',')
        self.max_context_num = max_context_num
        
        
        self.norm = self.data
        np.random.shuffle(self.norm)
        
        self.data_train = self.norm[:max_context_num]
        self.data_test = self.norm[max_context_num:]

    def train_data(self, diff):
        
        num_context = np.random.randint(self.max_context_num-diff,self.max_context_num) # extract random number of contexts

        num_target = self.max_context_num - num_context
        num_total_points = num_context + num_target # this num should be # of target points, = max_context_num
        total_id = np.random.choice(range(self.max_context_num), num_total_points, replace=False)
        context_id = total_id[:num_context]
        
        context_x = [[]]
        context_y = [[]]
        target_x = [[]]
        target_y = [[]]
        
        for i, id in enumerate(total_id):
            target_x[0].append(self.data_train[id,:1])
            target_y[0].append(self.data_train[id,1:])
            
        for i, id in enumerate(context_id):
            context_x[0].append(self.data_train[id,:1])
            context_y[0].append(self.data_train[id,1:])
            
        return np.array(context_x), np.array(context_y), np.array(target_x), np.array(target_y)
        
    def test_data(self):
        
        num_context = self.max_context_num
        num_target = 200 - num_context
        num_total_points = num_context + num_target # this num should be # of target points
        
        context_x = [[]]
        context_y = [[]]
        target_x = [[]]
        target_y = [[]]
        
        target_x[0] = np.concatenate((self.norm[:num_context,:1], self.norm[-num_target:,:1]), axis = 0)
        target_y[0] = np.concatenate((self.norm[:num_context,1:], self.norm[-num_target:,1:]), axis = 0)
            
        context_x[0] = self.data_train[:,:1]
        context_y[0] = self.data_train[:,1:]
        
        
        return np.array(context_x), np.array(context_y), np.array(target_x), np.array(target_y)
        
def plot_functions(target_x, target_y, context_x, context_y, pred_y, std, num_train, epoch, file_root, case):
    
    target_x = target_x.cpu().detach().numpy()
    target_y = target_y.cpu().detach().numpy()
    context_x = context_x.cpu().detach().numpy()
    context_y = context_y.cpu().detach().numpy()
    pred_y = pred_y.cpu().detach().numpy()
    std = std.cpu().detach().numpy()
  
    test_RMSE = np.sqrt(np.mean((pred_y[0, num_train:] - target_y[0, num_train:])**2))
    test_MAPE = np.mean(np.sqrt(np.square((pred_y[0, num_train:] - target_y[0, num_train:])\
                      /target_y[0, num_train:])))
    train_RMSE = np.sqrt(np.mean((pred_y[0, :num_train] - target_y[0, :num_train])**2))
    train_MAPE = np.mean(np.sqrt(np.square((pred_y[0, :num_train] - target_y[0, :num_train])\
                      /target_y[0, :num_train])))
    
    print('============================')
    print('test_RMSE: ', test_RMSE, 'test_MAPE: ', test_MAPE)
    print('train_RMSE: ', train_RMSE, 'train_MAPE: ', train_MAPE)
    print('============================')
    
    if epoch % 100 == 0:
        
        if case == "spiral":
    
            plt.scatter(pred_y[0,:,0], pred_y[0,:,1], c = 'b', marker = 'o')
            plt.scatter(target_y[0,:,0], target_y[0,:,1], c = 'k', marker = 'v')
            
            plt.errorbar(pred_y[0,:,0], pred_y[0,:,1], std[0,:,1]*1.96, std[0,:,0]*1.96, linestyle='None', capsize=12,
                     capthick=5, elinewidth=5, ecolor='r', label='Confidence Interval')

            plt.grid('off')
            ax = plt.gca()
            plt.savefig(file_root + '/spiral_test_%d.png' % (epoch))
            plt.clf()
        
        
        if case == "fea":
    
            plt.plot(pred_y[0], 'b', linewidth=2)
            plt.plot(target_y[0], 'k:', linewidth=2)
            plt.plot(context_y[0], 'ko', markersize=5)
            plt.fill_between(
                np.arange(pred_y.shape[1]),
                pred_y[0, :, 0] - std[0, :, 0],
                pred_y[0, :, 0] + std[0, :, 0],
                alpha=0.2,
                facecolor='#65c9f7',
                interpolate=True)

            plt.grid('off')
            ax = plt.gca()
            plt.savefig(file_root + '/fea_test_%d.png' % (epoch))
            plt.clf()
            
        
    return test_RMSE, test_MAPE, train_RMSE, train_MAPE         