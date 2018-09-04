#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code runs the wave equation forward in time. 

Example:
    
    python pytorch_wave_example.py -num_sources 10 -debug 1
    python pytorch_wave_example.py -h  (to see input parameters)

Created on Fri Jun 29 16:14:59 2018

@author: widemann1
"""

from utils import get_args, get_initial_conditions, run_wave_forward

#%%
def main():
    args = get_args()
    f,g = get_initial_conditions(args)
    u = run_wave_forward(f,g,args)
    return f,g, u

#%%
if __name__ == '__main__':

    f,g,u = main()
 




