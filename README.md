# Cardiac-MRI-Segmentation  

## Project Description  

This is my graduation project. Based on 3D-Unet, a 3D-segmentation network is implemented.

## Project Architecture

### MRI_data_read.py  

- test code for data read  
  
### dataset.py  

- dataloader for pytorch  
  
### dataset_generator.py  

- generate data for training and validation from raw medical images  
  
### loss_function.py  

- implementation of loss function used in training  
  
### main.py  

- contains some core function, train and predict  
  
### metrics.py  

- implementaion of metrics  
- it is slow, may need to optimize.  
  
### models.py  

- implementation of network model in pytorch  
  
### utli.py  

- some auxiliary function  
  
### test.py 

- implement a complete test program using the network  
- now finished.

## Current Work  

- now working on 3D unet training.  

## To Do  

- implement model_zoo.py for different models.  
- optimize metrics.py speed. 
- use decorator to get time tick
- transform3d.py for data augumation

## Further plans  

- multi-resolution enhancement  
- multi-model enhancement  
