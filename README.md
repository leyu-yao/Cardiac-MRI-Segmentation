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
  
### main_rev.py  

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

### timeTick.py  

- implement a decorator for time tick.  

### transform3d.py  

- implement class for transforms.  
- composed of tranforms is supported.  
- random select from transforms is supported.  

## Current Work  

- now working on 3D unet training.  

## To Do  

- implement model_zoo.py for different models.  
- optimize metrics.py speed. 
- transform3d.py for data augumation, enrich methods.  
- fix bug in test.py dynamicly select num_classes.  
- in np cutter, if stride is too large, may generate minus value in position.  
- add remap() in util.py remap [0,1,..7] to origin label value to visualize better.  

## Further plans  

- multi-resolution enhancement  
- multi-model enhancement  
