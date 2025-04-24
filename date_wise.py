from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
from loaddata_withindex import DLibdata
from torchsummary import summary
from model import ResNet, ResBlock, ResBottleneckBlock
import pandas as pd
import os
import scipy.io as sio
import warnings
import cv2
import numpy as np
from torchmetrics import R2Score
import argparse
from timeit import default_timer as timer
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
#from tensorboardX import SummaryWriter
from tqdm import tqdm
import utils
from openpyxl.workbook import Workbook
test_losses=[]
train_losses=[] 
output_data = []
target_data = []
index_values = []
index_t = []
def train(model,data_loader,optimizer, epoch):
    rl=0
    print('===> Training mode')
    output_data = torch.tensor([])
    target_data = torch.tensor([])
    num_batches = len(data_loader)
    total_step = epochs * num_batches
    epoch_tot_acc = 0
    model.train()
    if cuda_enabled:
        # When we wrap a Module in DataParallel for multi-GPUs
        model = model.module
    start_time = timer()
    for batch_idx, (data, target, index) in enumerate(tqdm(data_loader, unit='batch')):
        batch_size = data.size(0)
        global_step = batch_idx + (epoch * num_batches) - num_batches

        data, target, index = Variable(data), Variable(target), Variable(index)
        if cuda_enabled:
            data = data.cuda()
            target = target.cuda()
            index = index.cuda()
        torch.set_printoptions(sci_mode=False)
        optimizer.zero_grad()
        output = model(data) # output from DigitCaps (out_digit_caps)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     
        if batch_idx % 10 == 0:
            template = 'Epoch {}/{}, ' \
                    'Step {}/{}: ' \
                    '[Total loss: {:.4f}]' 
            tqdm.write(template.format(
                epoch,
                epochs,
                global_step,
                total_step,
                loss.item(),
                ))
       
        rl=rl+loss.item()

        if epoch == 100:
            index_values.extend(index.tolist())

            output_data = output_data.to(device)
            target_data = target_data.to(device)
            output_data = torch.cat((output_data, output), dim=0)
            target_data = torch.cat((target_data, target), dim=0)
            
    rl /= len(train_loader)
    train_losses.append(rl)     
    if epoch == 100:
        output_data_np = output_data.cpu().detach().numpy()
        target_data_np = target_data.cpu().detach().numpy() 
        index_df = pd.DataFrame(index_values, columns=['Index'])
        los = abs(output_data - target_data)
        plt.boxplot(los.detach().cpu().numpy(), labels=['B','Ca','Cu','Fe','K','Mg','Mn','Na','P','S','Zn'])

        # Set the title and axis labels
        plt.xlabel('parameters')
        plt.ylabel("MAE Values")
        plt.title('Box Plot for Training Dataset')
        plt.show()
        file_path = '/home/diksha/Images.txt'
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Create an empty list to store the fetched values
        new_df = pd.DataFrame()

        # Fetch the lines corresponding to the index values and concatenate the values
        for index_value in index_df['Index']:
            line = lines[index_value]  # Adjusting index value to match list indexing
            values = line.split('_')
            new_df = pd.concat([new_df, pd.DataFrame([values[0]])], ignore_index=True)

            #new_df = new_df.append(pd.Series(values[0]), ignore_index=True)
            
        output_df = pd.DataFrame(output_data_np, columns=['B','Ca','Cu','Fe','K','Mg','Mn','Na','P','S','Zn'])
        target_df = pd.DataFrame(target_data_np, columns=['B','Ca','Cu','Fe','K','Mg','Mn','Na','P','S','Zn'])

        output_df = pd.concat([new_df, index_df, output_df], axis=1)
        target_df = pd.concat([new_df, index_df, target_df], axis=1)

        output_excel_path = '/home/diksha/Documents/output.xlsx'
        target_excel_path = '/home/diksha/Documents/target.xlsx'

        with pd.ExcelWriter(output_excel_path) as writer:
            output_df.to_excel(writer, sheet_name='Output', index=False)

        with pd.ExcelWriter(target_excel_path) as writer:
            target_df.to_excel(writer, sheet_name='Target', index=False)

    end_time = timer()
    a=start_time-end_time
   
def imshow(img):
    img=img.cpu()
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  
    plt.show()

output_data_t = []
target_data_t = []
def tests(model,test_loader, epoch):
    print('===> Evaluate mode')
    output_data_t = torch.tensor([])
    target_data_t = torch.tensor([])
    model.eval()
    rl = 0
    num_batches = len(test_loader)
    total_step = epochs * num_batches
    for batch_idx, (data, target, index) in enumerate(tqdm(test_loader, unit='batch')):
        batch_size = data.size(0)
        global_step = batch_idx + (epoch * num_batches) - num_batches
        data, target, index = Variable(data), Variable(target), Variable(index)
        if cuda_enabled:
            data = data.cuda()
            target = target.cuda()
            index = index.cuda()
        torch.set_printoptions(sci_mode=False)
        
        output = model(data) # output from DigitCaps (out_digit_caps)
        loss = criterion(output, target)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        if batch_idx % 10 == 0:
            template = 'Epoch {}/{}, ' \
                    'Step {}/{}: ' \
                    '[Total loss: {:.4f}]' 
            tqdm.write(template.format(
                epoch,
                epochs,
                global_step,
                total_step,
                loss.item(),
                ))
        rl=rl+loss.item()
        
        if epoch == 100:
            output_data_t = output_data_t.to(device)
            target_data_t = target_data_t.to(device)
            output_data_t = torch.cat((output_data_t, output), dim=0)
            target_data_t = torch.cat((target_data_t, target), dim=0)
            index_t.extend(index.tolist())
            
    rl /= len(test_loader)
    test_losses.append(rl)
    #print("testloss",test_losses)
    if epoch == 100:
        output_data_t_np = output_data_t.cpu().detach().numpy()
        target_data_t_np = target_data_t.cpu().detach().numpy() 
        index_df = pd.DataFrame(index_t, columns=['Index'])
        file_path = '/home/diksha/Images_t.txt'

        # Open the file and read its contents
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Create an empty list to store the fetched values
        new_df = pd.DataFrame()
        los = abs(output_data_t - target_data_t)
        plt.boxplot(los.detach().cpu().numpy(), labels=['B','Ca','Cu','Fe','K','Mg','Mn','Na','P','S','Zn'])

        # Set the title and axis labels
        plt.xlabel('parameters')
        plt.ylabel("MAE Values")
        plt.title('Box Plot for Testing Dataset')
        plt.show()

        # Fetch the lines corresponding to the index values and concatenate the values
        for index_value in index_df['Index']:
            line = lines[index_value]  # Adjusting index value to match list indexing
            values = line.split('_')
            new_df = pd.concat([new_df, pd.DataFrame([values[0]])], ignore_index=True)
            #new_df = new_df.append(pd.Series(values[0]), ignore_index=True)

        output_df = pd.DataFrame(output_data_t_np, columns=['B','Ca','Cu','Fe','K','Mg','Mn','Na','P','S','Zn'])
        target_df = pd.DataFrame( target_data_t_np, columns=['B','Ca','Cu','Fe','K','Mg','Mn','Na','P','S','Zn'])

        output_df = pd.concat([new_df, index_df, output_df], axis=1)
        target_df = pd.concat([new_df, index_df, target_df], axis=1)

        output_excel_path = '/home/diksha/Documents/output_t.xlsx'
        target_excel_path = '/home/diksha/Documents/target_t.xlsx'

        with pd.ExcelWriter(output_excel_path) as writer:
            output_df.to_excel(writer, sheet_name='Output', index=False)

        with pd.ExcelWriter(target_excel_path) as writer:
            target_df.to_excel(writer, sheet_name='Target', index=False)


stt = timer()
cuda1=True
epochs=100
criterion = nn.L1Loss()

# Check GPU or CUDA is available
cuda_enabled = cuda1 and torch.cuda.is_available()
print(cuda_enabled)
kwargs = {'num_workers': 4,'pin_memory': True} if cuda_enabled else {}
training_set = DLibdata(train=True)  
print('===> Building model')        
train_loader = DataLoader(training_set, batch_size=32, shuffle=True, **kwargs)
model = ResNet(5, ResBottleneckBlock, [2, 2, 2, 2], useBottleneck=True, outputs=11)
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
warnings.filterwarnings("ignore")



if cuda_enabled:
    model.cuda()
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Make model checkpoint directory
if not os.path.exists('results/trained_model'):
    os.makedirs('results/trained_model')

# Train and test
tt=0.0
for epoch in range(1, epochs + 1):
    train(model, train_loader,optimizer, epoch)
    start_time = timer()
    # train your model
    if epoch == 100:
        testing_set = DLibdata(train=False)              
        test_loader = DataLoader(testing_set, batch_size=16, shuffle=True, **kwargs)
        tests(model,test_loader, epoch)
        end_time = timer()
        epoch_time = end_time - start_time
        tt=tt+epoch_time
        print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

    # Save model checkpoint
    utils.checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, epoch)

