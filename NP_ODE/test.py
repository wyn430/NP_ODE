from tqdm import tqdm
from network import LatentModel
from tensorboardX import SummaryWriter
import torchvision
import torch as t
from torch.utils.data import DataLoader
import os
import sys
from tools import *
import time
import matplotlib.pyplot as plt
import matplotlib


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def main(num_train, diff, dim, y_dim, case, noise_level):
    
    epochs = 30001
    num_hidden = 128
    input_dim = dim
    PLOT_AFTER = 1
    noise_levels = [0.01,0.02,0.1]
    path = './runs/' + model_dir + '/'
    
    if case == 'fea':
        dataset = MaterialFEADataset('../Data/data.txt', num_train)
        checkpoint = torch.load(path + 'MAPE_checkpoint.pth.tar')
    
    if case == 'spiral':
        spiral_filename = '../Data/' + 'spiral_data_range4_' + str(noise_levels[noise_level]) + '.txt'
        dataset = SpiralDataset(spiral_filename, num_train)
        checkpoint = torch.load(path + 'RMSE_checkpoint.pth.tar')
    
    
    model = LatentModel(num_hidden, input_dim, y_dim).cuda()
    
    model.load_state_dict(checkpoint['model'])
    context_x_test, context_y_test, target_x_test, target_y_test = dataset.test_data()
            
    context_x_test = torch.from_numpy(context_x_test).cuda()
    context_y_test = torch.from_numpy(context_y_test).cuda()
    target_x_test = torch.from_numpy(target_x_test).cuda()
    target_y_test = torch.from_numpy(target_y_test).cuda()

    # pass through the latent model
    t1 = time.time()
    y_pred, sigma, kl, loss = model(context_x_test, context_y_test, target_x_test)
    t2 = time.time()

    target_x = target_x_test.cpu().detach().numpy()
    target_y = target_y_test.cpu().detach().numpy()
    context_x = context_x_test.cpu().detach().numpy()
    context_y = context_y_test.cpu().detach().numpy()
    pred_y = y_pred.cpu().detach().numpy()
    std = sigma.cpu().detach().numpy()
  
    test_RMSE = np.sqrt(np.mean((pred_y[0, num_train:] - target_y[0, num_train:])**2))
    test_MAPE = np.mean(np.sqrt(np.square((pred_y[0, num_train:] - target_y[0, num_train:])\
                      /target_y[0, num_train:])))
    train_RMSE = np.sqrt(np.mean((pred_y[0, :num_train] - target_y[0, :num_train])**2))
    train_MAPE = np.mean(np.sqrt(np.square((pred_y[0, :num_train] - target_y[0, :num_train])\
                      /target_y[0, :num_train])))
    
    print('============================')
    print('test_RMSE: ', test_RMSE, 'test_MAPE: ', test_MAPE)
    print('train_RMSE: ', train_RMSE, 'train_MAPE: ', train_MAPE)
    print('time', t2-t1)
    print('============================')
    
    filename = './results/' + model_dir + '_results.npz'
    np.savez(filename, pred = pred_y[0,num_train:], target = target_y[0, num_train:], std = std[0,num_train:,0])
    #print(pred_y[0,num_train:].shape, target_y[0, num_train:].shape, std[0,num_train:,0].shape)
    
    '''
    labelfont = 30
    markerline = 25
    markerstar=17
    matplotlib.rc('xtick', labelsize=labelfont) 
    matplotlib.rc('ytick', labelsize=labelfont)
    
    if case == "spiral":
    
        plt.scatter(pred_y[0,:,0], pred_y[0,:,1], c = 'b', marker = 'o')
        plt.scatter(target_y[0,:,0], target_y[0,:,1], c = 'k', marker = 'v')

        plt.errorbar(pred_y[0,num_train:,0], pred_y[0,num_train:,1], std[0,num_train:,1], std[0,num_train:,0], linestyle='None', capsize=6, capthick=3, elinewidth=3, ecolor='r', label='Confidence Interval')

        plt.grid('off')
        ax = plt.gca()
        plt.savefig(path + '/test.png')
    
    if case == "fea":    
         
        plt.figure(figsize=(28,20))
        plt.plot(pred_y[0,num_train:,0], 'r_', markersize=markerline, mew=4, label='Prediction')
        plt.plot(target_y[0, num_train:], 'b*', markersize=markerstar, label='Ground Truth')
        plt.errorbar(np.arange(pred_y.shape[1]-num_train), pred_y[0,num_train:,0], std[0,num_train:,0]*1.96, linestyle='None', capsize=12,capthick=5, elinewidth=5, ecolor='c', label='Confidence Interval')
        plt.grid('off')
        plt.legend(fontsize=labelfont)
        ax = plt.gca()
        plt.ylim(-2.5,2.5)
        plt.xlim(-1,21)
        plt.xlabel('index of data points',fontsize=labelfont)
        # plt.ylabel('value of y',fontsize=labelfont)
        plt.savefig(path + 'test.png')
        
    '''
    
if __name__ == '__main__':
    num_train = int(sys.argv[1])
    diff = int(sys.argv[2])
    dim = int(sys.argv[3]) ## input dimension
    y_dim = int(sys.argv[4]) ## response dimension
    case = str(sys.argv[5]) ## two cases, spiral/fea
    noise_level = int(sys.argv[6]) ## level of noises in data 0,1,2
    model_dir = str(sys.argv[7])
    seed = 1
    
    torch.manual_seed(seed)
    main(num_train, diff, dim, y_dim, case, noise_level)