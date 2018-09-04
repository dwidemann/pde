#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 18:06:03 2018

@author: widemann1
"""

import os, sys
from time import time, sleep
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import argparse
import pprint as pp
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.animation as manimation


#%%
def get_args():
    parser = argparse.ArgumentParser(description='wave forward argparse',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-grid_size", type=int, default=100, help="create a grid of this many pixels")
    parser.add_argument("-time_steps", type=int, default=200, help="number of time steps to take")
    # CFL: dt <= C dx  (C = 1)
    parser.add_argument("-dt", type=float, default=.1, help="size of time step")
    parser.add_argument("-dx", type=float, default=.2, help="size of spatial step")
    parser.add_argument("-num_sources", type=int, default=1, help="number of sources to create")
    parser.add_argument("-debug", type=bool, default=False, help="visualizes output and vars")
    parser.add_argument("-inverse_debug", type=bool, default=False, help="visualizes output and vars for inverse")
    parser.add_argument("-lr", type=float, default=.001, help="learning rate")
    parser.add_argument("-max_u0", type=float, default=1., help="maximum starting value")
    parser.add_argument("-num_inv_steps", type=int, default=1000, help="number inverse steps")
    parser.add_argument("-write_fwd_movie", type=bool, default=False, help="write mp4 file")


    args = parser.parse_args()
    
    pp.pprint(vars(args))
    print('')

    return args
    

#%%
def plot_image(f,title='wave amplitude'):
    #plt.ion()
    u_mx = f.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)

#    cmap = cmap = sns.diverging_palette(250, 15, 
#                                        s=75, l=40, n=9,
#                                        as_cmap=True,center='dark') 
    cmap = plt.cm.ocean
    img = ax.imshow(f.data,cmap=cmap,vmin=-u_mx,vmax=u_mx)
    
    fig.colorbar(img, orientation='vertical')
    #plt.show()
    return img, fig
   
#%%
def Laplacian(u):
    u = u.unsqueeze(0).unsqueeze(0)
    L = np.array([[.5, 1, .5],[1, -6., 1],[.5, 1, .5]],'float32')
    pad = (L.shape[0]-1)//2
    conv = nn.Conv2d(1,1,L.shape[0],1,bias=False,padding=(pad,pad))
    conv.weight.data = torch.tensor(L).unsqueeze(0).unsqueeze(0)
    return conv(u).squeeze(0).squeeze(0)

#%%
def generate_input(num_sources=1,max_u0=1.,grid_size=100):
    u = torch.zeros((grid_size,grid_size),requires_grad=True)
    for src in range(num_sources):
        a,b = np.random.randint(0,grid_size,2)
        #u[grid_size//2, grid_size//2] = 1.
        u[a,b] = max_u0
    prev_u = torch.zeros((grid_size,grid_size),requires_grad=True)
    return Variable(u, requires_grad=True), Variable(prev_u, requires_grad=True)

#%%
def get_initial_conditions(args):
    u, prev_u = generate_input(args.num_sources,args.max_u0,args.grid_size)
    return u.clone(),prev_u.clone()

#%%
def run_wave_forward(u,prev_u,args):
    if args.debug:
        img, fig = plot_image(u)
        if args.write_fwd_movie:
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
            writer = FFMpegWriter(fps=15, metadata=metadata)
            writer.saving(fig, "forward_wave.mp4", args.time_steps)
            #writer.grab_frame()
        
    DT_DX_SQ = (args.dt/args.dx)**2
    for _ in range(args.time_steps):
        next_u = DT_DX_SQ*Laplacian(u) + 2*u - prev_u
        prev_u = u
        u = next_u
        #print(u.data.min())
        if args.debug:
            img.set_data(u.data)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(.0001)
            if args.write_fwd_movie:
                writer.grab_frame()
    return u

#%%
def run_heat_forward(u,args):
    if args.debug:
        img, fig = plot_image(u,'Diffusion')
    
    DT_DX_SQ = args.dt/(args.dx**2)
    for _ in range(args.time_steps):
        next_u = DT_DX_SQ*Laplacian(u) + u
        u = next_u
        if args.debug:
            img.set_data(u.data)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(.0001)

    return u


#%%
def run_heat_adjoint(u,args):
    if args.debug:
        img, fig = plot_image(u,'Adjoint Diffusion')
    
    DT_DX_SQ = args.dt/(args.dx**2)
    for _ in range(args.time_steps):
        next_u = -DT_DX_SQ*Laplacian(u) + u
        u = next_u
        if args.debug:
            img.set_data(u.data)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(.0001)

    return u    

#%%
def loss(u_est,u):
    err = ((u_est- u)**2).sum()
    return err


#%% scratch 

#x = torch.randn(3, requires_grad=True)
#
#y = x * 2
#while y.data.norm() < 1000:
#    y = y * 2
#
#print(y)
#
##%%
#gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
#y.backward(gradients)
#
#print(x.grad)


#%%
#def make_kernel(a):
#  """Transform a 2D array into a convolution kernel"""
#  a = np.asarray(a)
#  a = a.reshape(list(a.shape) + [1,1])
#  return tf.constant(a, dtype=1)
#
#def simple_conv(x, k):
#  """A simplified 2D convolution operation"""
#  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
#  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
#  return y[0, :, :, 0]
#
#def laplace(x):
#  """Compute the 2D laplacian of an array"""
#  laplace_k = make_kernel([[0.5, 1.0, 0.5],
#                           [1.0, -6., 1.0],
#                           [0.5, 1.0, 0.5]])
#  return simple_conv(x, laplace_k)

  

    
