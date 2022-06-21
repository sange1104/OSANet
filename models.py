from object_detection.faster_rcnn import PretrainedFasterRCNN
import torch.nn as nn
import torch
from tqdm import tqdm
import torchvision.models as models
import numpy as np 
 
class SelfAttention(nn.Module): 
    def __init__(self, glove_dim=300): 
        super().__init__() 
        self.glove_dim = glove_dim  
        self.W_1 = nn.Linear(self.glove_dim, 1) 
        self.W_2 = nn.Linear(self.glove_dim, self.glove_dim) 
        self.W_3 = nn.Linear(self.glove_dim, 1) 
        self.tanh = nn.Tanh() 

    def _get_weights(self,   
      values # (B, k, glove_dim) 
    ):   
        z = self.W_1(values) # key (batch_size, k, 1)
        weights = nn.functional.softmax(z, dim=1) # key (batch_size, k, 1)
        return weights
    
    def forward(self, 
      values 
    ):  
        weights = self._get_weights(values) # key (batch_size, k, 1) 
        new_ftr = torch.mul(weights, values) # (batch_size, k, glove_dim)
        new_ftr = new_ftr.sum(1) # (batch_size, glove_dim)
        return new_ftr, weights 
    
class LocalBranch:   
    def __init__(self, k = 10, glove_path = '../../EmojiGenerator/example/emoji-gan/utils/glove.6B.300d.txt'):
        self.k = k 
        self.glove_dim = 300 
        glove_dim = 300
        glove = self.load_glove(glove_path, glove_dim) 
        self.glove = glove
        self.fasterrcnn = PretrainedFasterRCNN() 
#         self.glove = load_glove(glove_path, glove_dim) 

    def load_glove(self, data_dir_path=None, embedding_dim=None):
        """
        Load the glove models (and download the glove model if they don't exist in the data_dir_path
        :param data_dir_path: the directory path on which the glove model files will be downloaded and store
        :param embedding_dim: the dimension of the word embedding, available dimensions are 50, 100, 200, 300, default is 300
        :return: the glove word embeddings (word -> embeddings)
        """
        if embedding_dim is None:
            embedding_dim = 300

        # glove_file_path = data_dir_path + "/glove.6B." + str(embedding_dim) + "d.txt"
        glove_file_path = data_dir_path
        # download_glove(data_dir_path, glove_file_path)
        _word2em = {}
        file = open(glove_file_path, mode='rt', encoding='utf8')
        for line in tqdm(file):
            words = line.strip().split()
            word = words[0]
            embeds = np.array(words[1:], dtype=np.float32)
            _word2em[word] = embeds
        file.close()
        return _word2em
        
    def _denormalize(self, img):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return np.array(img.cpu().permute(1,2,0)*255).astype(np.uint8)
        
    def get_objectembeddings(self, img): 
        '''
        img : (batch_size, c, h, w)
        '''
        word_emb_tensor = []
        for i in img:
            # 1) object detection
            classes = self.fasterrcnn.detect_object(self._denormalize(i)) 

            # 2) get word embeddings
            word_embs = torch.stack([torch.FloatTensor(self.glove[j]) if j in self.glove else torch.zeros(self.glove_dim) for j in classes])
            word_emb_tensor.append(word_embs) 
        return torch.stack(word_emb_tensor)


class OSANet(nn.Module):   
    def __init__(self, num_classes, use_pretrained=True):
        super(OSANet, self).__init__()  
        self.num_classes = num_classes
        feature_dim_1 = 2048
        feature_dim_2 = 64
        glove_dim = 300
        model_ft = models.resnet101(pretrained=use_pretrained)
        self.ftr_extrc = nn.Sequential(*list(model_ft.children())[:-1])
        self.att_list = nn.ModuleList([SelfAttention() for _ in range(self.num_classes)])
        self.lin_list = nn.ModuleList([nn.Linear(glove_dim, 1) for _ in range(self.num_classes)])
        self.k = 10
        self.dropout = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(feature_dim_1, glove_dim)
        self.linear_2 = nn.Linear(glove_dim, feature_dim_2)
        self.linear_3 = nn.Linear(self.k+1, 1)
        self.linear_4 = nn.Linear(1+glove_dim, self.num_classes)
        self.avgpool = nn.MaxPool1d(kernel_size=feature_dim_2)
        
    def forward(self, img, obj):
        # img: (batch, 3, 448, 448)
        # obj: (batch, 10, 300)
        
        # 1) Global branch
        g = self.ftr_extrc(img)    
         
        # 2) Semantic branch
        g_prime = self.linear_1(g.flatten(1)) # (batch, 300)
        g_prime = g_prime.unsqueeze(1) # (batch, 1, 300)
        
        weight_list = []
        b_list = []
        h = torch.cat([g_prime, obj], dim=1) # (batch, 11, 300)
        for i in range(self.num_classes):
            h_hat, weight = self.att_list[i](h) # (batch, 300) 
            h_hat = self.dropout(h_hat)
             
            b = self.lin_list[i](h_hat) # (batch, 1)  
            weight_list.append(weight)
            b_list.append(b)
        
        out = torch.cat(b_list, dim=1) # (batch, n_classes) 
        return out, weight_list