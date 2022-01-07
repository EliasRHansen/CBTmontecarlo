# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 02:52:19 2022

@author: Elias Roos Hansen
"""

import numpy as np
from electronsjumparound import carlo_CBT
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
kB=8.617*1e-5
e_SI=1.602*1e-19

N=100 #Number of islands
Ec=4.6e-6 #Charging energy in units of eV
Gt=2.16e-5 #Large voltage asymptotic conductance (affects noly the scaling of the result)
T=0.020 #Temperature in Kelvin
FWHM=5.439*kB*T*N #Full width half max according to the first order model
points=101 #number of voltages to run the simulation for
lim=1.5*FWHM 
V=np.linspace(-lim,lim,points)


####Run main simulation####
res=carlo_CBT(V,T,Ec,Gt,N=N,Nruns=10000,Ninterval=1000,Ntransient=10000,n_jobs=2,number_of_concurrent=8,
              parallelization='external',q0=0,dV=FWHM/50,batchsize=10)


####store main results###
mean_conductances=res.Gsm #mean conductance
std_conductance=res.Gstd #standard deviation of conductance
mean_currents=res.currentsm #mean currents
res.plotG(save=True)
res.plotI(save=True)
#%%
#######Crazy stuff####
from scipy.optimize import curve_fit
def load_data(idd_0,idd_1):
    try:
        dGs=[]
        voltages=[]
        currents=[]
        
        # idd_0=1954
        # idd_1=1956
        idds=np.array(range(idd_0,idd_1+1))
        path_data_arrays='C:\\Users\\Elias Roos Hansen\\Downloads\\SpinQubit-20211118T105030Z-001\\plots_thermometry(move)'
        try:
            for idd in idds:
                dG=np.load(path_data_arrays+'\\data_arrays\\dG_ID{}.npy'.format(idd))
                voltage=np.load(path_data_arrays+'\\data_arrays\\voltage_ID{:}.npy'.format(idd))
                current=np.load(path_data_arrays+'\\data_arrays\\current_ID{:}.npy'.format(idd))
                dGs.append(dG)
                voltages.append(voltage)
                currents.append(current)
        except FileNotFoundError:
            from qcodes.dataset.data_set import load_by_id #throws error if qcodes is not activated
            for j in np.arange(idd_0,idd_1+1):
        
                # idd_1=211
                current=load_by_id(j).get_parameter_data()['Ithaco']['Ithaco']
                dG=load_by_id(j).get_parameter_data()['Conductance']['Conductance']
                voltage=load_by_id(j).get_parameter_data()['Voltage']['Voltage']
                BNC20=load_by_id(j).get_parameter_data()['Voltage']['qdac_BNC20']
                np.save('C:\\Users\\Elias Roos Hansen\\Downloads\\SpinQubit-20211118T105030Z-001\\plots_thermometry(move)\\data_arrays\\dG_ID{}.npy'.format(j),dG)
                np.save('C:\\Users\\Elias Roos Hansen\\Downloads\\SpinQubit-20211118T105030Z-001\\plots_thermometry(move)\\data_arrays\\current_ID{}.npy'.format(j),current)
                np.save('C:\\Users\\Elias Roos Hansen\\Downloads\\SpinQubit-20211118T105030Z-001\\plots_thermometry(move)\\data_arrays\\voltage_ID{}.npy'.format(j),voltage)
                np.save('C:\\Users\\Elias Roos Hansen\\Downloads\\SpinQubit-20211118T105030Z-001\\plots_thermometry(move)\\data_arrays\\vBNC20_ID{}.npy'.format(j),BNC20)
            for idd in idds:
                dG=np.load(path_data_arrays+'\\data_arrays\\dG_ID{}.npy'.format(idd))
                voltage=np.load(path_data_arrays+'\\data_arrays\\voltage_ID{:}.npy'.format(idd))
                current=np.load(path_data_arrays+'\\data_arrays\\current_ID{:}.npy'.format(idd))
                dGs.append(dG)
                voltages.append(voltage)
                currents.append(current)

        lenvols=[]
        for vol in voltages:
            lenvols.append(len(vol))
        for v in np.arange(len(lenvols)):
            if lenvols[v]<np.max(lenvols):
                voltages.pop(v)
                currents.pop(v)
                dGs.pop(v)
        dGs=np.array(dGs)
        voltages=np.array(voltages)
        currents=np.array(currents)
        voltages_av=np.mean(voltages,axis=0)
        dGs_av=np.mean(dGs,axis=0)
        currents_av=np.mean(currents,axis=0)
        
        currents_std=np.std(currents,axis=0)/np.sqrt(len(idds))
        voltages_std=np.std(voltages,axis=0)/np.sqrt(len(idds))
        dGs_std=np.std(dGs,axis=0)/np.sqrt(len(idds))
        
        voltages_av=voltages_av[dGs_std>0]
        dGs_av=dGs_av[dGs_std>0]
        currents_av=currents_av[dGs_std>0]
        
        currents_std=currents_std[dGs_std>0]
        voltages_std=voltages_std[dGs_std>0]
        dGs_std=dGs_std[dGs_std>0]
        start=0
        voltages_av=voltages_av[start::]
        dGs_av=dGs_av[start::]
        currents_av=currents_av[start::]
        
        currents_std=currents_std[start::]
        voltages_std=voltages_std[start::]
        dGs_std=dGs_std[start::]
        return dGs,voltages,currents,dGs_std,voltages_std,currents_std,dGs_av,voltages_av,currents_av,idds
    except KeyError:
        voltages=[]
        currents=[]
        
        # idd_0=1954
        # idd_1=1956
        idds=np.array(range(idd_0,idd_1+1))
        path_data_arrays='C:\\Users\\Elias Roos Hansen\\Downloads\\SpinQubit-20211118T105030Z-001\\plots_thermometry(move)'
        try:
            for idd in idds:
                voltage=np.load(path_data_arrays+'\\data_arrays\\voltage_ID{:}.npy'.format(idd))
                current=np.load(path_data_arrays+'\\data_arrays\\current_ID{:}.npy'.format(idd))
                voltages.append(voltage)
                currents.append(current)
        except FileNotFoundError:
            from qcodes.dataset.data_set import load_by_id
            for j in np.arange(idd_0,idd_1+1):
        
                # idd_1=211
                current=load_by_id(j).get_parameter_data()['Ithaco']['Ithaco']
                voltage=load_by_id(j).get_parameter_data()['Voltage']['Voltage']

                np.save('C:\\Users\\Elias Roos Hansen\\Downloads\\SpinQubit-20211118T105030Z-001\\plots_thermometry(move)\\data_arrays\\current_ID{}.npy'.format(j),current)
                np.save('C:\\Users\\Elias Roos Hansen\\Downloads\\SpinQubit-20211118T105030Z-001\\plots_thermometry(move)\\data_arrays\\voltage_ID{}.npy'.format(j),voltage)

            for idd in idds:

                voltage=np.load(path_data_arrays+'\\data_arrays\\voltage_ID{:}.npy'.format(idd))
                current=np.load(path_data_arrays+'\\data_arrays\\current_ID{:}.npy'.format(idd))

                voltages.append(voltage)
                currents.append(current)
        lenvols=[]
        for vol in voltages:
            lenvols.append(len(vol))
        for v in np.arange(len(lenvols)):
            if lenvols[v]<np.max(lenvols):
                voltages.pop(v)
                currents.pop(v)

        voltages=np.array(voltages)
        currents=np.array(currents)
        # for s in np.arange(len(voltages)):
        #     voltages[s]=voltages[s]-np.mean(voltages[s])
        voltages_av=np.mean(voltages,axis=0)

        currents_av=np.mean(currents,axis=0)
        
        currents_std=np.std(currents,axis=0)/np.sqrt(len(idds))
        voltages_std=np.std(voltages,axis=0)/np.sqrt(len(idds))


        start=0
        voltages_av=voltages_av[start::]

        currents_av=currents_av[start::]
        
        currents_std=currents_std[start::]
        voltages_std=voltages_std[start::]

        return voltages,currents,voltages_std,currents_std,voltages_av,currents_av,idds


def fit_to_carlo(V,T,Ec,Gt,q0):
    dV=5.439*100*kB*T/30
    res=carlo_CBT(V,T,Ec,1,N=100,Nruns=2000,Ninterval=500,Ntransient=10000,number_of_concurrent=8,
                  parallelization='external',q0=q0,dV=dV)
    return Gt*res.Gsm
idd_0=1855
idd_1=1859

dGs,voltages,currents,dGs_std,voltages_std,currents_std,dGs_av,voltages_av,currents_av,idds=load_data(idd_0,idd_1)

V_data=voltages_av[400:1100][::10]
G_data=dGs_av[400:1100][::10]
plt.figure()
plt.plot(V_data,G_data,'.')
p0=[30e-3,4e-6,2.17e-5,0]
res=curve_fit(fit_to_carlo,V_data,G_data,p0=p0,epsfcn=5e-3,full_output=True)
#%%


kB=8.617*1e-5
e_SI=1.602*1e-19

N=100 #Number of islands
Ec=4.6e-6 #Charging energy in units of eV
Gt=2.16e-5 #Large voltage asymptotic conductance (affects noly the scaling of the result)
T=0.020 #Temperature in Kelvin
FWHM=5.439*kB*T*N #Full width half max according to the first order model
points=101 #number of voltages to run the simulation for
lim=1.5*FWHM 
# V=np.linspace(-lim,lim,points)
V=voltages_av[400:1100][::10]
G_data=dGs_av[400:1100][::10]
Chi_sq=[]
G_mins=[]
Chi_mins=[]
Ecs=[4.0e-6,4.25e-6,4.5e-6,4.75e-6]
Ts=[16e-3,18e-3,20e-3,22e-3]
####Run main simulation####
from itertools import product
combinations=product(Ecs,Ts)

for Ec,T in combinations:
    print(Ec)
    print(T)
    res=carlo_CBT(V,T,Ec,Gt,N=N,Nruns=9000,Ninterval=1000,Ntransient=10000,n_jobs=2,number_of_concurrent=8,
                  parallelization='external',q0=0,dV=FWHM/50,batchsize=10)


    ####store main results###
    mean_conductances=res.Gsm #mean conductance
    std_conductance=res.Gstd #standard deviation of conductance
    mean_currents=res.currentsm #mean currents
    res.plotG(save=True)
    plt.plot(V_data,G_data,'.',label='experimental data, BF cooldown 1')
    plt.legend()
    res.plotI(save=True)
    res.savedata()
    chi_squares=[]
    G=np.linspace(2e-5,3e-5,5000)
    for i in np.arange(len(G)):
        chi_squares.append(np.sum((G_data-G[i]*mean_conductances/Gt)**2/(G[i]*std_conductance/Gt)**2))
    
    chi_min=min(chi_squares)
    index_min=chi_squares.index(chi_min)
    G_min=G[index_min]
    print('G_min: '+str(G_min))
    chi_squares=np.array(chi_squares)
    
    fig,ax=plt.subplots()
    ax.plot(G,chi_squares,label='$\chi^2_0={:.1f}'.format(chi_min))
    ax.set_ylabel(r'$\chi^2$')
    ax.set_xlabel('Tunneling conductances [Si]')
    plt.legend()
    fig.savefig(res.filepath+'Chi_sq_plot.png')
    Chi_sq.append(chi_squares)
    Chi_mins.append(chi_min)
    G_mins.append(G_min)
    plt.close()
    