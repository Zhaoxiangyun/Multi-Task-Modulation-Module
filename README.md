# A Modulation Module for Multi-task Learning with Applications in Image Retrieval


This is a TensorFlow implementation of our ECCV paper
["A Modulation Module for Multi-task Learning with Applications in Image Retrieval"](https://arxiv.org/abs/1807.06708). 
The released code include the training code for 7 attributes. This code could support training with any number of training attributes. The code can be easily changed for 20 and 40 number of attributes training. We have released the prepared training and testing data for 7 attributes. The training data and testing data for 20 attributes training will come soon.   
## Requirement
The code is tested using Tensorflow 1.0.0 under Ubuntu 14.04 with Python 3.5. The code is based on the facenet [impelementation](https://github.com/davidsandberg/facenet). You could refer it for the software requirements.  

## Training and testing data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training and testing. This training set consists of around ranmdomly selected 30,000 image pairs over 40 attributes after face detection. We have prepared the training and testing data [here]().


## Running training and testing
./experiments/train_mm.sh
./experiments/test_mm.sh

## Performance
You could find the trained model [here]
