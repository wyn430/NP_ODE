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
import time
torch.manual_seed(2)

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def main(num_train, diff, dim, y_dim, case, noise_level):
    
    epochs = 40001
    num_hidden = 128
    input_dim = dim
    y_dim = y_dim
    PLOT_AFTER = 100
    noise_levels = [0.01,0.02,0.1]
    
    if case == 'fea':
        dataset = MaterialFEADataset('../Data/data.txt', num_train)
        folder_prefix = 'fea_range4'
    
    if case == 'spiral':
        spiral_filename = '../Data/' + 'spiral_data_range4_' + str(noise_levels[noise_level]) + '.txt'
        dataset = SpiralDataset(spiral_filename, num_train)
        folder_prefix = 'spiral_range4_' + str(noise_levels[noise_level])
        
    par_root = "./runs"
    file_root = par_root + '/' + folder_prefix + '_' + time.strftime("%Y%m%d-%H%M%S")
    file_name = file_root + '/log.txt'
    image_root = file_root + '/imgs'
    os.mkdir(file_root)
    os.mkdir(image_root)
    
    model = LatentModel(num_hidden, input_dim, y_dim).cuda()
    model.train()
    
    optim = t.optim.Adam(model.parameters(), lr=1e-4)
    
    global_step = 0
    test_MAPE = 10000
    test_RMSE = 10000
    
    info = "num of context: " + str(num_train) + " diff " + str(diff)
    with open(file_name, 'a') as f:
        f.write(info+'\n')
    
    for epoch in range(epochs):
        t1 = time.time()
        
        context_x_train, context_y_train, target_x_train, target_y_train = dataset.train_data(diff)
        
        if epoch % PLOT_AFTER == 0:
            context_x_test, context_y_test, target_x_test, target_y_test = dataset.test_data()
            
            context_x_test = torch.from_numpy(context_x_test).cuda()
            context_y_test = torch.from_numpy(context_y_test).cuda()
            target_x_test = torch.from_numpy(target_x_test).cuda()
            target_y_test = torch.from_numpy(target_y_test).cuda()
            
            # pass through the latent model
            y_pred, sigma, kl, loss = model(context_x_test, context_y_test, target_x_test)
            
            errors = plot_functions(target_x_test, target_y_test, context_x_test, context_y_test, y_pred, sigma, num_train, epoch, image_root, case)
            
            
                
             # save model by each epoch 
            if test_MAPE > errors[1]:
                log = 'Epoch: ' + str(epoch) + ' test_RMSE: ' + str(errors[0]) + ' test_MAPE: ' + str(errors[1]) + ' train_RMSE: ' + str(errors[2]) + ' train_MAPE: ' + str(errors[3])
                with open(file_name, 'a') as f:
                    f.write(log+'\n')
                    
                test_MAPE = errors[1]
                t.save({'model':model.state_dict(),'optimizer':optim.state_dict()},
                       os.path.join(file_root,'MAPE_checkpoint.pth.tar'))
#                t.save({'model':model.state_dict(),'optimizer':optim.state_dict()},
#                       os.path.join(file_root,'checkpoint_%d.pth.tar' % (epoch+1)))
            elif test_RMSE > errors[0]:
                log = 'Epoch: ' + str(epoch) + ' test_RMSE: ' + str(errors[0]) + ' test_MAPE: ' + str(errors[1]) + ' train_RMSE: ' + str(errors[2]) + ' train_MAPE: ' + str(errors[3])
                with open(file_name, 'a') as f:
                    f.write(log+'\n')
                    
                test_RMSE = errors[0]
                t.save({'model':model.state_dict(),'optimizer':optim.state_dict()},
                       os.path.join(file_root,'RMSE_checkpoint.pth.tar'))

            
        
        global_step += 1
        adjust_learning_rate(optim, global_step)
        context_x_train = torch.from_numpy(context_x_train).cuda()
        context_y_train = torch.from_numpy(context_y_train).cuda()
        target_x_train = torch.from_numpy(target_x_train).cuda()
        target_y_train = torch.from_numpy(target_y_train).cuda()

        # pass through the latent model
        y_pred, sigma, kl, loss = model(context_x_train, context_y_train, target_x_train, target_y_train)

        # Training step
        optim.zero_grad()
        loss.backward()
        optim.step()
        t2 = time.time()
        log = 'Epoch: ' + str(epoch) + ' training_loss: ' + ' loss: ' + str(loss.cpu().detach().numpy()) + ' kl: ' + str(kl.mean().cpu().detach().numpy())
        print(log)
        print('time: ', t2-t1)
        with open(file_name, 'a') as f:
          f.write(log+' time: '+ str(t2-t1) + '\n')
        
       
if __name__ == '__main__':
    num_train = int(sys.argv[1])
    diff = int(sys.argv[2])
    dim = int(sys.argv[3]) ## input dimension
    y_dim = int(sys.argv[4]) ## response dimension
    case = str(sys.argv[5]) ## two cases, spiral/fea
    noise_level = int(sys.argv[6]) ## level of noises in data 0,1,2,3,4
    main(num_train, diff, dim, y_dim, case, noise_level)