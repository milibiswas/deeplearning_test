# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:40:57 2019

DataLoader : Messey Dataset

@author: Mili Biswas
"""
#########################  Module Import ###########################

import os
import shutil as sh
import numpy as np
from torchvision.transforms import Compose,ToTensor,Resize,Normalize,RandomHorizontalFlip,RandomRotation
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



#######################  User defined functions ####################


class data_loader_messey(object):
    
    def __init__(self,SourceDatasetPath="../data/dataset/messeydataset",pathTestDataSource="../test/dataset/messeydataset"):
        self.SourceDatasetPath=SourceDatasetPath
        self.path = "../data/tmp_messey"
        self.path1 = "../data"
        self.path_valid = "../data/valid"
        self.path_train = "../data/train"
        self.pathTestDataSource=pathTestDataSource
        self.pathTestDataTarget="../test/testdata"
            

        ###################   Necessary Directory Creation ################
    
        if not os.path.exists(self.SourceDatasetPath):
            print("Source directory :"+self.SourceDatasetPath+" does not exists, so exiting.")
            exit()
            
        if not os.path.exists(self.pathTestDataSource):
            print("Test data directory :"+self.pathTestDataSource+" does not exists, so exiting.")
            exit()
            
        if os.path.exists(self.path):
            sh.rmtree(self.path)
            os.mkdir(self.path)
            self.multiple_file_copy(self.SourceDatasetPath,self.path)
        else:
            os.mkdir(self.path)
            self.multiple_file_copy(self.SourceDatasetPath,self.path)
            
            
        if os.path.exists(os.path.join(self.path1,'valid')):
            sh.rmtree(os.path.join(self.path1,'valid'))
            
        if os.path.exists(os.path.join(self.path1,'train')):
            sh.rmtree(os.path.join(self.path1,'train'))
            
        os.mkdir(os.path.join(self.path1,'valid'))
        os.mkdir(os.path.join(self.path1,'train'))
        
        if os.path.exists(self.pathTestDataTarget):
            sh.rmtree(self.pathTestDataTarget)
        
        os.mkdir(self.pathTestDataTarget)
        
        ############## Calling to Data Preparation Functions ##################
        # The shuffle will hold image files' indexes in random order
        
        self.shuffle = np.random.permutation(len(os.listdir(self.path)))
        self.ls = []
        
        # preparing valid folder test/valid
        for i in os.listdir(self.path):
            self.ls.append((i.split('_')[1],i,))
        
        self.prepare_test_data(self.ls,0,600)
        self.prepare_valid_data(self.ls,600,1000)
        self.prepare_train_data(self.ls,1000)
        
        
        # preparing dataset-train dataset/ validation datadset
        self.train_transform = Compose([Resize([64,64]),RandomHorizontalFlip(0.2),ToTensor(),Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
        self.simple_transform = Compose([Resize([64,64]),ToTensor(),Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
        self.train_dataset = ImageFolder(self.path_train,transform=self.train_transform)
        self.valid_dataset = ImageFolder(self.path_valid,transform=self.simple_transform)
        print(self.pathTestDataTarget)
        self.test_dataset=ImageFolder(self.pathTestDataTarget,transform=self.simple_transform)
        #print(valid_dataset[0])
        # preparing dataloader - train dataloader /validation dataloader
        
        self.train_dataloader = DataLoader(self.train_dataset,batch_size=5)
        self.valid_dataloader = DataLoader(self. valid_dataset,batch_size=5)
        self.test_dataloader = DataLoader(self. test_dataset,batch_size=5)
        
        ################ Removing temporary paths #######################
        
        sh.rmtree(self.path)
        
        
        ######################  End of Constructor ######################
    
    def multiple_file_copy(self,src_path,dst_path):
        for i in os.listdir(src_path):
            sh.copy2(os.path.join(src_path,i),dst_path)
    
    def prepare_valid_data(self,ls,initIndex,endIndex):
        print("Validation data preparation phase")
        for i in self.shuffle[initIndex:endIndex]:
            if os.path.exists(os.path.join(self.path_valid,ls[i][0])):
                os.rename(os.path.join(self.path,ls[i][1]),os.path.join(self.path_valid,ls[i][0],ls[i][1]))
            else:
                os.mkdir(os.path.join(self.path_valid,ls[i][0]))
                os.rename(os.path.join(self.path,ls[i][1]),os.path.join(self.path_valid,ls[i][0],ls[i][1]))
    
    def prepare_train_data(self,ls,initIndex,endIndex=-1):
        print("Train data preparation phase")
        for i in self.shuffle[initIndex:endIndex]:
            if os.path.exists(os.path.join(self.path_train,ls[i][0])):
                os.rename(os.path.join(self.path,ls[i][1]),os.path.join(self.path_train,ls[i][0],ls[i][1]))
            else:
                os.mkdir(os.path.join(self.path_train,ls[i][0]))
                os.rename(os.path.join(self.path,ls[i][1]),os.path.join(self.path_train,ls[i][0],ls[i][1]))
                
    def prepare_test_data(self,ls,initIndex,endIndex):
        tmp_path=os.path.join(self.pathTestDataTarget,'test_tmp')
        print("Test data preparation phase")
        if os.path.exists(tmp_path):
            sh.rmtree(tmp_path)
            os.mkdir(tmp_path)
        else:
            os.mkdir(tmp_path)
            
        for i in self.shuffle[initIndex:endIndex]:
            if os.path.exists(os.path.join(tmp_path,ls[i][0])):
                os.rename(os.path.join(self.path,ls[i][1]),os.path.join(tmp_path,ls[i][0],ls[i][1]))
            else:
                os.mkdir(os.path.join(tmp_path,ls[i][0]))
                os.rename(os.path.join(self.path,ls[i][1]),os.path.join(tmp_path,ls[i][0],ls[i][1]))

        self.pathTestDataTarget=tmp_path
        
        
        
    
    '''def prepare_test_data(self,pathTestDataSource,pathTestDataTarget):
        ls=[]
        tmp_path=os.path.join(pathTestDataTarget,'test_tmp')
        print(tmp_path)
        if os.path.exists(tmp_path):
            sh.rmtree(tmp_path)
            os.mkdir(tmp_path)
            self.multiple_file_copy(pathTestDataSource,tmp_path)
        else:
            os.mkdir(tmp_path)
            self.multiple_file_copy(pathTestDataSource,tmp_path)
        
        
        for i in os.listdir(tmp_path):
            ls.append((i.split('_')[1],i,))
    
        for i,j in enumerate(ls):
            if os.path.exists(os.path.join(pathTestDataTarget,ls[i][0])):
                os.rename(os.path.join(tmp_path,ls[i][1]),os.path.join(pathTestDataTarget,ls[i][0],ls[i][1]))
            else:
                os.mkdir(os.path.join(pathTestDataTarget,ls[i][0]))
                os.rename(os.path.join(tmp_path,ls[i][1]),os.path.join(pathTestDataTarget,ls[i][0],ls[i][1]))
        
        sh.rmtree(tmp_path)'''
    

if __name__=="__main__":
    dl=data_loader_messey()
