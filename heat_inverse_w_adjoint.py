#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this worked:
    
 python heat_inverse_w_adjoint.py -inverse_debug 1 -dx 1 -time_steps 20 -num_sources 20
 python heat_inverse_w_adjoint.py -h
 
Note: 
    1. It finds the blurred sources in 1 step. 
    2. It doesn't work if time_steps is too great, e.g. 50. 

Created on Fri Jun 29 16:14:59 2018

@author: widemann1
"""
from utils import get_args, get_initial_conditions, run_heat_forward, loss,\
plot_image, run_heat_adjoint
import matplotlib.pyplot as plt

#%%
def main(args):
    # run the model forward with some initial conditions 
    # these are the labels. 
    f,g = get_initial_conditions(args)
    u = run_heat_forward(f,args)

    f_est = run_heat_adjoint(u,args)

    f_err = loss(f_est,f) #(((f-f_est)**2).sum()
    print('f_err: {:.4f}'.format(f_err))
    u_est = run_heat_forward(f_est,args)
    err = ((u_est- u.data)**2).sum()
    print('u_T error: {:.4f}'.format(err))
       
#    for _ in range(args.num_inv_steps):
#        f_est = run_heat_adjoint(u,args)
#        
#        # compute the error and gradients
##        err = ((f_est- f.data)**2).sum()
##        f_est.register_hook(save_grad('f_est'))
##        err.backward()
##        f_est = f_est - args.lr*grads['f_est']
##        print('u_T error: {:.4f}'.format(err))
#        f_err = loss(f_est,f) #(((f-f_est)**2).sum()
#        print('f_err: {:.4f}'.format(f_err))
#        u_est = run_heat_forward(f_est,args)
#        err = ((u_est- u.data)**2).sum()
#        print('u_T error: {:.4f}'.format(err))
#    
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
 





