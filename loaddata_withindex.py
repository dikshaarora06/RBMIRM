from __future__ import print_function
#import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from torchvision import transforms
import csv
import torch._utils
import pandas as pd
to_tensor = transforms.Compose([ transforms.ToTensor()])

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
class DLibdata:
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    
    def __init__(self, train=True):
        
        self.train = train  # training set or test set

        if self.train:
            txt_file = open("/home/diksha/Images.txt")
            self.img_names = []
            for i in txt_file:
                i = i.strip("\n")
                i = '/home/diksha/Crop_by_field/new/'+i
                self.img_names.append(i)
                
            self.results = []
            k=0
            with open("/home/diksha/LeafSampleAnalysis.csv") as csvfile: 
                 reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                 for row in reader:     
                     k=k+1    
                     self.results.append(row)
            
        else:
            txt_file_t = open("/home/diksha/Images_t.txt")
            self.img_names_t = []
            for i in txt_file_t:
                i = i.strip("\n")
                i = '/home/diksha/Crop_by_field/new/'+i
                self.img_names_t.append(i)
                
            self.results_t = []
            k=0
            with open("/home/diksha/LeafSampleAnalysis_t.csv") as csvfile: 
                 reader_t = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                 for row in reader_t:     
                     k=k+1    
                     self.results_t.append(row)
                

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            h_values = []
            image5 = torch.empty(0)
            image_n = self.img_names[index]
            image_n = image_n.split(".")
            h = image_n[0]
            h_values.append(h)
            channel=['red','green','blue','red_edge','nir']
            for k in channel:
                 i=Image.open(image_n[0]+'.'+image_n[1]+k+'.tif').convert('L')
                 chn = to_tensor(i)
                 image5=torch.cat((image5,chn),0)   
            b = self.results[index]
            arr = torch.FloatTensor(b)
            return image5, arr, index
            #img, target = self.train_data[index], self.train_labels[index]
        else:
            image5_t = torch.empty(0)
            channel=['red','green','blue','red_edge','nir']
            for k in channel:
                 image_n_t = self.img_names_t[index]
                 image_n_t = image_n_t.split(".")
                 i=Image.open(image_n_t[0]+'.'+image_n_t[1]+k+'.tif').convert('L')
                 chn_t = to_tensor(i)
                 image5_t=torch.cat((image5_t,chn_t),0)
            b_t= self.results_t[index]
            arr_t = torch.FloatTensor(b_t)
            return image5_t, arr_t, index
        
    def __len__(self):
        if self.train:
            return len(self.img_names)
        else:
            return len(self.img_names_t)

 