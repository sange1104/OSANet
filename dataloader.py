from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import os
from PIL import Image
import torch
import numpy as np
from sklearn import preprocessing

class dataset(Dataset):
    def __init__(self, data_name, le, mode, data_dir):
        data_dir_path=  os.path.join(data_dir, data_name, mode)
        self.file_list = [i for emo in os.listdir(data_dir_path) for i in os.listdir(os.path.join(data_dir_path, emo))]
        self.path_list = [os.path.join(data_dir_path, os.path.join(i.split('_')[0], i)) for i in self.file_list] 
        if data_name == 'flickr' or data_name == 'FI':
            self.name_list = [int(i.split('.')[-2].split('_')[-1]) for i in self.file_list]
        elif data_name == 'instagram':
            self.name_list = [int(i.split('.')[-2].split('_')[-2]) for i in self.file_list] 
        self.label_list = [le.transform([i.split('/')[-2]])[0] for i in self.path_list]  
        self.mode = mode    
        self.pos_emotion = ['amusement', 'contentment', 'awe', 'excitement']
        self.neg_emotion = ['anger', 'disgust', 'fear', 'sadness'] 
        
        input_size = 448
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def __len__(self):
        return len(self.path_list) 
    
    def __getitem__(self, item):   
        image_path = self.path_list[item]
        image_name_idx = self.name_list[item] 
        image = Image.open(image_path).convert("RGB")
        # resize image
        image = self.transforms[self.mode](image) 
        
        label = self.label_list[item]              
    
        return { 
          'idx' : torch.tensor(image_name_idx, dtype=torch.int64) ,
          'image' : image,  
          'label': torch.tensor(label, dtype=torch.int64) 
        }

def get_label_encoder(dir):
    label_list = []
    for i in os.listdir(dir):
        label_list.append(i.split('_')[0])
    label_list = list(set(label_list))

    le = preprocessing.LabelEncoder()
    le.fit(label_list)
    return le

def load_dataloader(data_name, le, batch_size, mode, data_dir, num_workers=2): 
    ds = dataset(data_name = data_name, le=le, mode=mode, data_dir=data_dir) 
    data_loader = DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle = True, num_workers=num_workers)
    return data_loader