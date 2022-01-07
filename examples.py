# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 02:52:19 2022

@author: Elias Roos Hansen
"""

import numpy as np
from final_version_WIP import carlo_CBT
kB=8.617*1e-5
e_SI=1.602*1e-19

N=100 #Number of islands
Ec=4e-6 #Charging energy in units of eV
Gt=2e-5 #Large voltage asymptotic conductance (affects noly the scaling of the result)
T=0.01 #Temperature in Kelvin
FWHM=5.439*kB*T*N #Full width half max according to the first order model
points=101 #number of voltages to run the simulation for
lim=2.5*FWHM 
V=np.linspace(-lim,lim,points)


####Run main simulation####
res=carlo_CBT(V,T,Ec,Gt,N=N,Nruns=100000,Ninterval=1000,Ntransient=10000,n_jobs=4,number_of_concurrent=20,
              parallelization='external',q0=0)


####store main results###
mean_conductances=res.Gsm #mean conductance
std_conductance=res.Gstd #standard deviation of conductance
mean_currents=res.currentsm #mean currents
res.plotG(save=True)
res.plotI(save=True)