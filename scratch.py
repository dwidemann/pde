#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:04:36 2018

@author: widemann1
"""
import argparse
import os, sys
from time import time, sleep
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

#%%
parser = argparse.ArgumentParser(description="Do something.")
parser.add_argument('-x', '--x-center', default=.1,type=float)
parser.add_argument('-y', '--y-center', type=float, default=.1)
parser.add_argument('-inputDims','--inputDims', default=100)
parser.add_argument('values', type=float, nargs='*')
parser.add_argument("-debug", type=bool, default=False, help="visualizes output and vars")
args = parser.parse_args()

for k,v in args.__dict__.items():
    print('{}: {}'.format(k,v))
    

#%%

#parser = argparse.ArgumentParser(description='wave forward argparse',
#                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
#parser.add_argument("grid_size", type=int, default=100, help="create a grid of this many pixels")
#parser.add_argument("time_steps", type=int, default=200, help="number of time steps to take")
#parser.add_argument("dt", type=float, default=.1, help="size of time step")
#parser.add_argument("dx", type=float, default=.1, help="size of spatial step")
#parser.add_argument("num_sources", type=int, default=1, help="number of sources to create")
##parser.add_argument("debug", type=bool, default=False, help="visualizes output and vars")
#args = parser.parse_args()
#
#pp.pprint(vars(args))
#    
#%%
x = torch.randn(3, requires_grad=True)

y = x * 2
#while y.data.norm() < 1000:
#    y = y * 2

print(y)

#%%
#gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
gradients = torch.ones(3,dtype=torch.float)
y.backward(gradients)

print(x.grad)

#%%
x = torch.randn(1,requires_grad=True)
f = 2*x + 1
f.backward()
x.grad

#%%
x = torch.randn((2,2),requires_grad=True)
f = 2*x + 1

label = torch.randn((2,2))
#err = nn.MSELoss()(f,label)
err = ((f - label)**2).sum()
err.backward()
print(x.grad)

#%%
lr = .01
label = torch.randn((2,2))
x = Variable(torch.randn((2,2)),requires_grad=True)
for i in range(100):
    f = 2*x + 1
    err = ((f - label)**2).sum()
    print(err)
    err.backward()
    x = x - lr*x.grad

#%%
lr = .001
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

x = Variable(torch.randn(1,1), requires_grad=True)
label = torch.randn(1,1)
for i in range(1000):
    y = 3*x
    z = ((y-label)**2).sum()

    # In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
    x.register_hook(save_grad('x'))
    y.register_hook(save_grad('y'))
    z.register_hook(save_grad('z'))
    z.backward()
    x = x - lr*grads['x']
    print(z)
#    print(grads['x'])
#    print(grads['y'])
#    print(grads['z'])    
#%%
lr = .001
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

x = Variable(torch.randn(1,1), requires_grad=True)
label = torch.randn(1,1)
for i in range(1000):
    for i in range(10):
        y = 3*x
    
    z = ((y-label)**2).sum()

    # In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
    x.register_hook(save_grad('x'))
    y.register_hook(save_grad('y'))
    z.register_hook(save_grad('z'))
    z.backward()
    x = x - lr*grads['x']
    print(z)
#    print(grads['x'])
#    print(grads['y'])
#    print(grads['z'])    

#%%
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

x = Variable(torch.randn(1,1), requires_grad=True)
y = 3*x
z = y**2

# In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
y.register_hook(save_grad('y'))
z.register_hook(save_grad('z'))
z.backward()

print(grads['y'])
print(grads['z'])    

#%%
optimizer = optim.SGD(err, lr=0.001, momentum=0.9)
err.backward()
optimizer.step()


#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(err, lr=0.001, momentum=0.9)


#%%
optimizer.zero_grad()

# forward + backward + optimize
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
    