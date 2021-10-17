from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import shutil, argparse, json

local_machine=True
# Func to load saved model
def load_model(path):
    if local_machine:
        checkpoint= torch.load(path, map_location='cpu')
    else:
        checkpoint= torch.load(path)
    model=neural_net
    
    for params in model.parameters():
        params.requires_grad= False
    
    return model

#Func to process loaded image from database
def process_image(image):
    img_transforms=transforms.Compose(
        [
        transforms.Resize((48,48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    image=img_transforms(Image.open(image))
    
    return image

#Func to display loaded image after transforms have been applied
def display_img(image, ax=None, title=None):
    if ax is None:
        fig,ax=plt.subplots()
        
    image=image.numpy().transpose((1,2,0))
        
    mean= np.array([0.5])
    std=np.array([0.5])
        
    image= std*image+mean
        
    ax.display_img(image)
    
    return ax
# func to transcribe text from image using loaded model
def transcribe(image_path, neural_net):
    image_data= process_image(image_path)
    model=load_model(path)
    model_p= model.eval()
    inputs= Variable(image_data.unsqueeze(0))
    output=model_p(inputs)
    
    return output


if __name__== '__main__':
    main()