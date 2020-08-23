# SegmentRoads
Semantic Segmentation of Roads From Satellite Imagery

This repo has the code of two attempts to solve the semantic segmentation of roads from satellite imagery.

First attempt is to train a semantic segmentation network using the AWS sagemaker inbuilt algorithm. 

The second is inspired from skyeyenet and uses a custom trained network to do semantic segmentation of the satellite imagery for road detection.

In the second approach images were cropped from the original images and masks to resize them to 256*256.  
This allows for better training efficiency. 

Activation function used was elu - Exponential Linear Unit as it helps converge faster. The HE Normal initializer is used for a truncated normal distribution centered around 0. It is noticed that some image quality is lost in translation from jpeg to png.
The neural network architecture was taken from skyeyenet with 5 upsampling layers and 4 downsampling layers to map to the masks images.

The data is stored on 

https://segmentskyeye.s3.amazonaws.com/

The model may be described as below
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 256, 256, 3)  0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 256, 256, 16) 448         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 256, 256, 16) 64          conv2d_1[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 256, 256, 16) 0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 256, 256, 16) 2320        dropout_1[0][0]                  
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 256, 256, 16) 64          conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 128, 128, 16) 0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 128, 128, 32) 4640        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 128, 128, 32) 128         conv2d_3[0][0]                   
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 128, 128, 32) 0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 128, 128, 32) 9248        dropout_2[0][0]                  
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 128, 128, 32) 128         conv2d_4[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 64, 64, 32)   0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 64, 64, 64)   18496       max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 64, 64, 64)   256         conv2d_5[0][0]                   
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 64, 64, 64)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 64, 64, 64)   36928       dropout_3[0][0]                  
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 64, 64, 64)   256         conv2d_6[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 32, 32, 64)   0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 128)  73856       max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 128)  512         conv2d_7[0][0]                   
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 32, 32, 128)  0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 128)  147584      dropout_4[0][0]                  
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 32, 32, 128)  512         conv2d_8[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 16, 16, 128)  0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 16, 256)  295168      max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 16, 16, 256)  1024        conv2d_9[0][0]                   
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 16, 16, 256)  0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 16, 256)  590080      dropout_5[0][0]                  
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 16, 16, 256)  1024        conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 32, 32, 128)  131200      batch_normalization_10[0][0]     
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 32, 32, 256)  0           conv2d_transpose_1[0][0]         
                                                                 batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 128)  295040      concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 32, 32, 128)  512         conv2d_11[0][0]                  
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 32, 32, 128)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 128)  147584      dropout_6[0][0]                  
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 32, 32, 128)  512         conv2d_12[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 64, 64, 64)   32832       batch_normalization_12[0][0]     
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 64, 64, 128)  0           conv2d_transpose_2[0][0]         
                                                                 batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 64, 64, 64)   73792       concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 64, 64, 64)   256         conv2d_13[0][0]                  
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 64, 64, 64)   0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 64, 64, 64)   36928       dropout_7[0][0]                  
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 64, 64, 64)   256         conv2d_14[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 128, 128, 32) 8224        batch_normalization_14[0][0]     
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 128, 128, 64) 0           conv2d_transpose_3[0][0]         
                                                                 batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 128, 128, 32) 18464       concatenate_3[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 128, 128, 32) 128         conv2d_15[0][0]                  
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 128, 128, 32) 0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 128, 128, 32) 9248        dropout_8[0][0]                  
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 128, 128, 32) 128         conv2d_16[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_4 (Conv2DTrans (None, 256, 256, 16) 2064        batch_normalization_16[0][0]     
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 256, 256, 32) 0           conv2d_transpose_4[0][0]         
                                                                 batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 256, 256, 16) 4624        concatenate_4[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 256, 256, 16) 64          conv2d_17[0][0]                  
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 256, 256, 16) 0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 256, 256, 16) 2320        dropout_9[0][0]                  
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 256, 256, 16) 64          conv2d_18[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 256, 256, 1)  17          batch_normalization_18[0][0]     
==================================================================================================
Total params: 1,946,993
Trainable params: 1,944,049
Non-trainable params: 2,944
__________________________________________________________________________________________________

An early stopper callback is implemented to stop the training if there is no change in validation loss for 5 epochs consecutively.

The loss function chosen is the dice loss. The dice loss tries to maximise the overlap between the output image and groundtruth image. 
The loss function is based on the dice coeffecient where which ranges from 0 to 1 where a Dice coefficient of 1 denotes perfect and complete overlap.

The results from this model are shared in the notebook itself. It is observed that it does reasonably well with the urban settings. 
This attempt was done as a side project and given more time, the best iteration would be to use the png format for training as converting to jpeg introduces noise in the masks.

To build the full roads these predicted masks can be stitched and some postprocessing algorithms may be developed for interpolation of results.

Another direction to investigate would be to try Semantic Segmentation For Single Class as we are interested in only roads class. Such an attempt may be found at
https://medium.com/analytics-vidhya/pytorch-implementation-of-semantic-segmentation-for-single-class-from-scratch-81f96643c98c

This may be used as an inspiration to do semantic segmentation with one class as input.


