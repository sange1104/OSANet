from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import argparse
import os
from utils import EmoLoss, load_yaml
from models import OSANet, LocalBranch
from tqdm import tqdm
from dataloader import load_dataloader, get_label_encoder 

class Trainer(): 
    def __init__(self, device_num):
#         self.args = args 
        self.device = self.set_device(device_num)
        self.set_loss_fn()
        self.num_epochs = 200
        self.learning_rate = 3e-05
        self.weight_decay = 1e-05
        self.num_classes = 8
        self.alpha = 0.5
        self.hist = defaultdict(list)
        self.save_dir = './checkpoints'
        self.best_acc = 0.
        self.scheduling = True
        self.data_name = 'FI'
        self.loss_writer = SummaryWriter('logs/loss')
        self.acc_writer = SummaryWriter('logs/acc')

    def set_device(self, device_num):
        if torch.cuda.is_available(): 
            print('There are %d GPU(s) available.'%torch.cuda.device_count())
            print('We will use the GPU: %d'%device_num)
            return torch.device("cuda:0")
        else: 
            print('No GPU available, using the CPU instead.')
            return torch.device("cpu")
    
    def get_params(self, model, feature_extract=False):
        params_to_update = model.parameters() 
        if feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param) 
        return params_to_update

    def set_optimizer(self, params): 
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50], gamma=0.5)
        return optimizer, scheduler
    
    def set_loss_fn(self):
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.emo_loss_fn = EmoLoss() 

    def build_model(self):
        model = OSANet(self.num_classes).to(self.device)
        localbranch = LocalBranch()
        return model, localbranch
        
    def compute_loss(self, outputs, preds, labels):
        # cross-entropy loss
        ce_loss = self.ce_loss_fn(outputs, labels)
        
        # emotional loss
        emo_loss = self.emo_loss_fn(preds, labels)
        
        # total loss
        total_loss = ce_loss * (1 - self.alpha) + emo_loss * self.alpha
        return total_loss
        
    def backward_step(self, loss, optimizer, scheduler):
        # calculate loss
        loss.backward()
        
        # update params
        optimizer.step()
        if self.scheduling:
            scheduler.step()
            
        # initialize
        optimizer.zero_grad()
        
    def compute_acc(self, preds, labels):
        assert len(preds) == len(labels)
        return torch.sum(preds.data == labels.data) / len(preds)
    
    def _init_epoch_info(self): 
        total_preds = []
        total_labels = []
        total_losses = 0.
        return total_preds, total_labels, total_losses
        
    def _cum_epoch_info(self, preds, label, loss, total_preds, total_labels, total_losses): 
        total_preds.append(preds)
        total_labels.append(label)
        total_losses += loss.item()
        return total_preds, total_labels, total_losses
    
    def _cum_hist(self, train_loss, train_acc, val_loss, val_acc): 
        self.hist['train_acc'].append(train_acc) 
        self.hist['train_loss'].append(train_loss)
        self.hist['val_acc'].append(val_acc) 
        self.hist['val_loss'].append(val_loss)
        
    def save_checkpoint(self, model):
        filename = os.path.join(self.save_dir, '%s_%.2f.pt'%(self.data_name, self.best_acc))
        torch.save(model.state_dict(), filename)
        print("Saving Model to:", filename, "...Finished.")

    def predict(self, model, branch, dataloader): 
        model.eval() 
        total_preds, total_labels, total_losses = self._init_epoch_info()
        
        for batch in tqdm(dataloader):
            image = batch['image'].to(self.device) 
            labels = batch['label'].to(self.device) 

            # forward
            obj = branch.get_objectembeddings(image).cuda()
            outputs, _ = model(image, obj) 
            _, preds = torch.max(outputs, 1) 

            # compute loss
            loss = self.compute_loss(outputs, preds, labels)
            
            # cumulate
            total_preds, total_labels, total_losses = self._cum_epoch_info(preds, labels, loss, total_preds, total_labels, total_losses)
    
        total_losses /= len(total_preds)
        total_preds = torch.cat(total_preds, 0)
        total_labels = torch.cat(total_labels, 0)
        return total_losses, total_preds, total_labels
            
    def evaluate(self, model, branch, dataloader):
        loss, preds, labels = self.predict(model, branch, dataloader)
        acc = self.compute_acc(preds, labels)
        return loss, acc
        
    def train(self):
        ################
        # 1. Load data
        ################   
        le = get_label_encoder(os.path.join(DATA_DIR, DATA_NAME, 'train'))
        train_data_loader = load_dataloader(DATA_NAME, le, BATCH_SIZE, 'train', DATA_DIR) 
        val_data_loader = load_dataloader(DATA_NAME, le, BATCH_SIZE, 'val', DATA_DIR) 
        test_data_loader = load_dataloader(DATA_NAME, le, BATCH_SIZE, 'test', DATA_DIR) 
        
        #####################
        # 2. Build a model
        ##################### 
        model, branch = self.build_model()
        params = self.get_params(model)
        optimizer, scheduler = self.set_optimizer(params)
        
        #####################
        # 3. Train
        ##################### 
        for it in range(self.num_epochs): 
            model.train()                  
            total_preds, total_labels, total_losses = self._init_epoch_info()
            
            for batch in tqdm(train_data_loader):
                image = batch['image'].to(self.device) 
                labels = batch['label'].to(self.device) 

                # forward
                obj = branch.get_objectembeddings(image).cuda()
                outputs, _ = model(image, obj)
                _, preds = torch.max(outputs, 1) 

                # compute loss
                loss = self.compute_loss(outputs, preds, labels) 
                 
                # backward
                self.backward_step(loss, optimizer, scheduler)

                # cumulate 
                total_preds, total_labels, total_losses = self._cum_epoch_info(preds, labels, loss, total_preds, total_labels, total_losses)
            
            
            train_loss = total_losses/len(total_preds)
            total_preds = torch.cat(total_preds, 0)
            total_labels = torch.cat(total_labels, 0)
            train_acc = self.compute_acc(total_preds, total_labels)
            
            # evaluate model performance for validation set
            val_loss, val_acc = self.evaluate(model, branch, val_data_loader)
            
            print(f'Epoch {it + 1}/{self.num_epochs}')
            print('-' * 10)
            print(f'Train loss {train_loss} Train acc {train_acc}')
            print(f'Val   loss {val_loss}   Val acc {val_acc}')
            self._cum_hist(train_loss, train_acc, val_loss, val_acc) 
            self.loss_writer.add_scalars("Loss", {'train' : train_loss, 'val' : val_loss}, it) 
            self.acc_writer.add_scalars("Acc", {'train' : train_acc, 'val' : val_acc}, it)

            # save model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(model)
                
        print('-'*10)
        print('End Training at Epoch %d!'%self.num_epochs)
        test_loss, test_acc = self.evaluate(model, test_data_loader)
        print(f'Test   loss {test_loss}   Test acc {test_acc}')
        

if __name__ == '__main__': 
    ######################################
    # configuration
    ###################################### 
    PROJECT_DIR = os.path.dirname(__file__)
    train_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'train_config.yaml'))
    DATA_NAME = train_config['DATALOADER']['data_name']
    DATA_DIR = train_config['DATALOADER']['dir_path']
    BATCH_SIZE = train_config['DATALOADER']['batch_size']
    
    GPU_NUM = train_config['GPU_NUM']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%GPU_NUM

    
    trainer = Trainer(GPU_NUM)
    trainer.train()

    