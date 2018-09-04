#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python pytorch_wave_example_inverse.py  -inverse_debug 1 -lr .001 -time_steps 20 -num_sources 10

Created on Fri Jun 29 16:14:59 2018

@author: widemann1
"""
from utils import get_args, get_initial_conditions, run_wave_forward, loss,\
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
    u = run_wave_forward(f,g,args)
    
    # creat new initial condtions
    f_est, g_est = get_initial_conditions(args)
    
    if args.inverse_debug:
        img, fig = plot_image(f_est,title='Estimated Sources')
        
    for _ in range(args.num_inv_steps):
        u_est = run_wave_forward(f_est,g_est,args)
        
        # compute the error and gradients
        err = ((u_est- u.data)**2).sum()
        f_est.register_hook(save_grad('f_est'))
        err.backward()
        f_est = f_est - args.lr*grads['f_est']
        print('u_T error: {:.4f}'.format(err))
        f_err = loss(f_est,f) #(((f-f_est)**2).sum()
        print('f_err: {:.4f}'.format(f_err))
        if args.inverse_debug:
            img.set_data(f_est.data)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(.0001)
            
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
 





