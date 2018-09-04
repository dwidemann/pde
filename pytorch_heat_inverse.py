#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this worked:
    
    python pytorch_heat_inverse.py -num_inv_steps 1000 -inverse_debug 1 -dx 1 -lr 1. -time_steps 3
    
Note: When the time_steps is big, e.g. 100, there is no error because we are doing diffusion. So,
finding the inverse does not get enough error signal. 

Created on Fri Jun 29 16:14:59 2018

@author: widemann1
"""
from utils import get_args, get_initial_conditions, run_heat_forward, loss,\
plot_image
import matplotlib.pyplot as plt

#%%

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

#%%
def main(args):
    # run the model forward with some initial conditions 
    # these are the labels. 
    f,g = get_initial_conditions(args)
    u = run_heat_forward(f,args)
    
    # creat new initial condtions
    f_est, g_est = get_initial_conditions(args)
    
    for _ in range(args.num_inv_steps):
        u_est = run_heat_forward(f_est,args)
        
        # compute the error and gradients
        err = ((u_est- u.data)**2).sum()
        f_est.register_hook(save_grad('f_est'))
        err.backward()
        f_est = f_est - args.lr*grads['f_est']
        print('u_T error: {:.4f}'.format(err))
        f_err = loss(f_est,f) #(((f-f_est)**2).sum()
        print('f_err: {:.4f}'.format(f_err))
    
    if args.inverse_debug:
        img, fig = plot_image(f,title='True Initial Condition (f)') 
        plt.figure(1)
        img, fig = plot_image(f_est,title='Estimated Initial Condition (f_est)')
        plt.figure(2)
        plt.show()
    return f, f_est
        
                

#%%
if __name__ == '__main__':
    args = get_args()
    f, f_est = main(args)
 





