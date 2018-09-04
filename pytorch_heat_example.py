#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code runs the wave equation forward in time. 

Example:
    
    python pytorch_heat_example.py -num_sources 10 -dt .1 -dx 1. -debug 1
    python pytorch_heat_example.py -h  (to see input parameters)

Created on Fri Jun 29 16:14:59 2018

@author: widemann1
"""

from utils import get_args, get_initial_conditions, run_heat_forward

#%%
def heat_forward(args):
    
    f,g = get_initial_conditions(args)
    u = run_heat_forward(f,args)
    return f, u

#%%
if __name__ == '__main__':
    args = get_args()
    f,u = heat_forward(args)
 




