import torch
import numpy as np
import yaml

class EmoLoss(torch.nn.Module):
    '''calculate emotional loss using Mikel's Wheel distance''' 
    def __init__(self):
        super(EmoLoss, self).__init__()
        self.idx_to_emo = {0:'amusement', 1:'anger', 2:'awe', 3:'contentment', 4:'disgust', 5:'excitement', 6:'fear', 7:'sadness'}
        self.emo_to_idx_loss = {'fear':0, 'excitement':1, 'awe':2, 'contentment':3, 'amusement':4, 'anger':5, 'disgust':6, 'sadness':7}

    def _convert_to_new_idx(self, input_list):
        '''convert to new idx which indicates the position in the Mikel's Wheel'''
        emo_list = [self.idx_to_emo[int(i.item())] for i in input_list]
        new_idx_list = [self.emo_to_idx_loss[e] for e in emo_list]
        return torch.LongTensor(new_idx_list)
    
    def _get_min_dist(self, pred, true):
        dist1 = torch.abs(pred - true)
        dist2 = torch.abs(pred + 8 - true)
        dist3 = torch.abs(true + 8 - pred)
        
        total_dist = torch.stack((dist1, dist2, dist3))
        min_dist = torch.min(total_dist, dim=0)[0].float()
        return min_dist
    
    def forward(self, logits, labels):
        '''
        # Arguments
         - logits : (batch_size, 1)
         - labels : (batch_size, 1)
        '''
        assert logits.size() == labels.size()
        new_pred_list = self._convert_to_new_idx(logits)
        new_true_list = self._convert_to_new_idx(labels)
        min_dist = self._get_min_dist(new_pred_list, new_true_list)
        
        return torch.mean(min_dist)

def denormalize(img):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return np.array(img.cpu().permute(1,2,0)*255).astype(np.uint8)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)