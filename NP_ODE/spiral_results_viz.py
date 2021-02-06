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
    model_files = ['spiral_range4_0.01', 
                   'spiral_range4_0.02', 
                   'spiral_range4_0.1']
    
    if case == 'fea':
        dataset = MaterialFEADataset('../Data/data.txt', num_train)
        folder_prefix = 'fea_range4'
    
    if case == 'spiral':
        spiral_filename = '../Data/' + 'spiral_data_range4_' + str(noise_levels[noise_level]) + '.txt'
        dataset = SpiralDataset(spiral_filename, num_train)
        folder_prefix = 'spiral_range4_' + str(noise_levels[noise_level])
    
    path = './runs/' + model_files[noise_level] + '/'



    model = LatentModel(num_hidden, input_dim, y_dim).cuda()
    checkpoint = torch.load(path + 'RMSE_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['model'])
    context_x_test, context_y_test, target_x_test, target_y_test = dataset.test_data()
            
    context_x_test = torch.from_numpy(context_x_test).cuda()
    context_y_test = torch.from_numpy(context_y_test).cuda()
    target_x_test = torch.from_numpy(target_x_test).cuda()
    target_y_test = torch.from_numpy(target_y_test).cuda()

    # pass through the latent model
    y_pred, sigma, kl, loss = model(context_x_test, context_y_test, target_x_test)

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
    #print(std)
    print('============================')
    
    #filename = '../results/50_NP_ODE.npz'
    #np.savez(filename, pred = pred_y[0,num_train:], target = target_y[0, num_train:], std = std[0,num_train:,0])
    #print(pred_y[0,num_train:].shape, target_y[0, num_train:].shape, std[0,num_train:,0].shape)
    
    labelfont = 50
    legendfont = 50
    markerline = 25
    markerstar=17
    linewidth = 5
    matplotlib.rc('xtick', labelsize=labelfont) 
    matplotlib.rc('ytick', labelsize=labelfont)
    spiral_wonoise = np.loadtxt("../Data/spiral_data_range4.txt", dtype='float32',delimiter=',')
    
    if case == "spiral":

        f = plt.figure(figsize=(80,20))
        ax1 = f.add_subplot(411)
        ax2 = f.add_subplot(412)
        ax3 = f.add_subplot(413)
        ax4 = f.add_subplot(414)
    
        result = np.concatenate((target_x,pred_y,std), axis = -1)[0]
        result = result[result[:,0].argsort()]
        
        ax1.plot(spiral_wonoise[:,1], spiral_wonoise[:,2], lw=linewidth, c = 'b', label = "spiral")
        ax1.scatter(target_y[0,:num_train,0], target_y[0,:num_train,1], c = 'g', s = 400, marker = '*', label = "training data")
        ax1.scatter(target_y[0,num_train:,0], target_y[0,num_train:,1], c = 'r', s = 300, marker = 's', label = "testing data")
        ax1.set_xlabel('y1', fontsize=labelfont)
        ax1.set_ylabel('y2', fontsize=labelfont)
        
        
        ax2.plot(spiral_wonoise[:,1], spiral_wonoise[:,2], lw=linewidth, c = 'b', label = "spiral")
        ax2.plot(result[:,1], result[:,2], lw=linewidth, c = 'r', label = "generation")
        ax2.fill_between(result[:,1], result[:,2] + result[:,4], result[:,2] - result[:,4], 
                        facecolor='yellow', alpha=0.8)
        ax2.set_title('Generated spiral with UQ in y2', fontsize=labelfont)
        ax2.set_xlabel('y1', fontsize=labelfont)
        ax2.set_ylabel('y2', fontsize=labelfont)
        
        ax3.plot(spiral_wonoise[:,1], spiral_wonoise[:,2], lw=linewidth, c = 'b', label = "spiral")
        ax3.plot(result[:,1], result[:,2], lw=linewidth, c = 'r', label = "generation")
        ax3.fill_betweenx(result[:,2], result[:,1] + result[:,3], result[:,1] - result[:,3], 
                        facecolor='cyan', alpha=0.8)
        ax3.set_title('Generated spiral with UQ in y1', fontsize=labelfont)
        ax3.set_xlabel('y1', fontsize=labelfont)
        ax3.set_ylabel('y2', fontsize=labelfont)
        
        ax4.plot(spiral_wonoise[:,1], spiral_wonoise[:,2], lw=linewidth, c = 'b', label = "spiral")
        ax4.plot(result[:,1], result[:,2], lw=linewidth, c = 'r', label = "generation")
        ax4.fill_between(result[:,1], result[:,2] + result[:,4], result[:,2] - result[:,4], 
                        facecolor='yellow', alpha=0.8)
        ax4.fill_betweenx(result[:,2], result[:,1] + result[:,3], result[:,1] - result[:,3], 
                        facecolor='cyan', alpha=0.8)
        ax4.set_title('Generated spiral with UQ in y1 and y2', fontsize=labelfont)
        ax4.set_xlabel('y1', fontsize=labelfont)
        ax4.set_ylabel('y2', fontsize=labelfont)
#         plt.scatter(result[:,1], result[:,1], c = 'b', marker = 'o')
        

#         plt.errorbar(pred_y[0,num_train:,0], pred_y[0,num_train:,1], std[0,num_train:,1], std[0,num_train:,0], linestyle='None', capsize=6, capthick=3, elinewidth=3, ecolor='r', label='Confidence Interval')

        ax1.legend(fontsize=legendfont, loc='lower right')
        ax2.legend(fontsize=legendfont, loc='lower right')
        ax3.legend(fontsize=legendfont, loc='lower right')
        ax4.legend(fontsize=legendfont, loc='lower right')
        plt.grid('off')
        plt.gca()
#         plt.show()
        plt.savefig('./results/spiral_result_' + str(noise_levels[noise_level]) + '.png',bbox_inches='tight',pad_inches=0)
    
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
      
    
    return std
    
    
if __name__ == '__main__':
    num_train = 150
    diff = 30
    dim = 3 ## input dimension
    y_dim = 2 ## response dimension
    case = 'spiral' ## two cases, spiral/fea
    noise_level = int(sys.argv[1])
    seed = 1
    
    torch.manual_seed(seed)
    std = main(num_train, diff, dim, y_dim, case, noise_level)