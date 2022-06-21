# OSANet
This repository contains the implementation code for the paper "OSANet: Object Semantic Attention Network for Visual Sentiment Analysis".

<!-- Abstract
------------
Visual sentiment analysis, which aims to predict human emotional response to visual stimuli, has attracted considerable attention with the increasing popularity of sharing images online. Most studies have focused on improving emotion recognition using holistic and local information derived from given images. Little attention has been paid to semantic information of objects in images, which plays a significant role in human emotional response to images. In this study, we propose a novel object semantic attention network (OSANet), which attempts to unravel the semantic information of the image objects that contribute to emotion detection. OSANet combines both global representation and semantic information of objects to predict the emotion corresponding to a given image. While holistic features which represent the entire image are extracted by convolutional blocks, the objectlevel semantic information is first obtained from pretrained word embedding and then weighted according to the relative importance of the object with the attention mechanism. Notably, a new loss function to deal with the subjectivity of the sentiment analysis is introduced, which improves the performance of emotion detection task. Extensive experiments with three image emotion datasets demonstrate the superiority and interpretability of OSANet. The results show that OSANet achieves the greater performance than other image emotion detection frameworks. -->

Overview😎
------------
- [data/]() is the top level data directory. Here we assume it consists of different kinds of dataset. Also, each dataset folder is divided into train, validation and test set folder. The data directory is expected to consist as follows.

```bash
data
├── FI
│   ├── train
│   ├── val
│   └── test
├── flickr
│   ├── train
│   ├── val
│   └── test
└── instagram
    ├── train
    ├── val
    └── test
```


Model Architecture
-------------
![model_architecture_v3](https://user-images.githubusercontent.com/63252403/160062419-98627eef-b131-4dcd-a835-af1ab50d72c2.png)

Setup
---------------------------
**Environment setup**

For experimental setup, requirements.txt lists down the requirements for running the code on the repository. Note that a cuda device is required. The requirements can be downloaded using,

```
pip install -r requirements.txt
``` 

Usage
------------------------
1. Clone the repository
    ```
    git clone https://github.com/sange1104/OSANet.git
    ```

2. Download dataset and split into train, val, and test set.

3. Set up the object_detection folder. We used pre-trained object detection model from the [Faster R-CNN with model pretrained on Visual Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome). For the object_detection folder, you better follow the guideline of [this repository](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome).

4. Train the model with arguments from [/config/train_config.yaml](https://github.com/sange1104/OSANet/blob/main/config/train_config.yaml). 
    You can train the model as follows:
    ```
    python train.py
    ```

5. The checkpoints of the best validation performance will be saved in [/checkpoints]() directory. You can further train model or use for inference with this checkpoint.
