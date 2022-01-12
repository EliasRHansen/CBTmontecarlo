# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 02:52:19 2022

@author: Elias Roos Hansen
"""

import numpy as np
from electronsjumparound import carlo_CBT,fit_carlo
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
# res=curve_fit(fit_to_carlo,V_data,G_data,p0=p0,epsfcn=5e-3,full_output=True)
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
#%%
#crazy stuff that actually works

kB=8.617*1e-5
e_SI=1.602*1e-19

N=100 #Number of islands
# Ec=4.6e-6 #Charging energy in units of eV
# Gt=2.16e-5 #Large voltage asymptotic conductance (affects noly the scaling of the result)
# T=0.020 #Temperature in Kelvin
# FWHM=5.439*kB*T*N #Full width half max according to the first order model
points=131 #number of voltages to run the simulation for
# lim=1.5*FWHM 
# V=np.linspace(-lim,lim,points)
V_data=voltages_av#[300:1200]
V_data_std=voltages_std#[300:1200]
G_data=dGs_av#[300:1200]
G_data_std=dGs_std#[300:1200]

####Run main simulation####
from itertools import product
from scipy.interpolate import interp1d
import os





def chi(a,b,delta):
    return np.sum((a-b)**2/delta**2)
# q0s=np.linspace(-0.2,0.2,3)
unitless_u=np.linspace(2,2.6,10)
chi_sq_uniform=[]
chi_sq_0=[]
for u in unitless_u:

        # print(u)

        lim=5.2*5.439*N
        V=np.linspace(-lim,lim,points)
        number_of_concurrent=20
        q0=0*np.random.uniform(low=-1,high=1,size=(N-1,))
        Nruns=30000
        Ninterval=10
        Ntransient=300000
        transient=500
        offset_C=0*np.ones((N-1,))/10
        =0#np.random.uniform(low=-5e-1,high=5e-1,size=(N,))
        second_order_C=np.ones((N,))/10
        res=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                      parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
    
    
        ####store main results###
        mean_conductances=res.Gsm #mean conductance
        std_conductance=res.Gstd #standard deviation of conductance
        mean_currents=res.currentsm #mean currents
        model=interp1d(V,mean_conductances,kind='linear',bounds_error=False,fill_value=(np.mean(mean_conductances[0:3]),np.mean(mean_conductances[-3::])))
        wacky_sigma=interp1d(V,std_conductance,kind='linear',bounds_error=False,fill_value=(np.mean(std_conductance[0:3]),np.mean(std_conductance[-3::])))
        
        def f0(V,Ec,Gt,V0,T):
            return Gt*(1-(Ec/(kB*T))*res.CBT_model_g((V_data-V0)/(N*kB*T)))
        p0=[4e-6,2.16e-5,0,30e-3]
        par0,cov0=curve_fit(f0,V_data,G_data,p0=p0)
        def f(V,Ec,Gt,V0):
            return Gt*model((V-V0)/Ec)
                
        p_model=[par0[0],par0[1],par0[2]]
        error_check=True
        number_of_tries=0
        max_tries=5
        while error_check==True:
            number_of_tries+=1
            try:
                
                par,cov=curve_fit(f,V_data,G_data,p0=p_model,sigma=wacky_sigma(V_data))
                print(par)
                G_MC=f(V_data,*par)
                chi_model=chi(G_data,G_MC,wacky_sigma(V_data/par[0])*par[1]/np.sqrt(number_of_concurrent))
                chi_0=chi(G_data,f0(V_data,*par0),np.mean(wacky_sigma(V_data/par[0])*par[1]/np.sqrt(number_of_concurrent)))
                chi_sq_uniform.append(chi_model)
                fig=plt.figure(figsize=(11,6))
                plt.errorbar(V_data,G_data,fmt='.',label='experimental data',yerr=G_data_std,xerr=V_data_std)
                plt.title('Best MC Fit parameters for u={:.2f}, q0="uniformly distributed": '.format(u)+' T={:.1f} mK'.format(1e3*par[0]/(u*kB))+'\n $G_T={:.1e}$'.format(par[1])+r' $\Omega^{-1}$'+' $E_c$={:.1e} $\mu$eV, $C_0$/C={:.2f}, <$\delta C^2$>/C={:.2f}, $C^(2)$/C={:.3f}'.format(1e6*par[0],np.mean(offset_C),np.mean(dC**2),np.mean(second_order_C)))
                # plt.errorbar(V_data,G_MC,yerr=wacky_sigma(V_data/par[0])*par[2]/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')
                plt.errorbar(V*par[0]+par[2],mean_conductances*par[1],yerr=par[1]*std_conductance/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')
                plt.ylabel('conductance [Si]')
                plt.xlabel('Bias voltage [V]')
                plt.tight_layout()
                
                plt.legend(loc=3)
                try:
                    fig.savefig(res.filepath+'Chi_sq_plot1.png')
                except FileNotFoundError:
                    os.mkdir(res.filepath)
                    fig.savefig(res.filepath+'Chi_sq_plot1.png')
                    
                fig=plt.figure(figsize=(11,6))
                plt.errorbar(V_data,G_data,fmt='.',label='experimental data',yerr=G_data_std,xerr=V_data_std)
                plt.title('Best MC Fit parameters for u={:.2f}, q0="uniformly distributed": '.format(u)+' T={:.1f} mK'.format(1e3*par[0]/(u*kB))+'\n $G_T={:.1e}$'.format(par[1])+r' $\Omega^{-1}$'+' $E_c$={:.1e} $\mu$eV'.format(1e6*par[0]))
                # plt.errorbar(V_data,G_MC,yerr=wacky_sigma(V_data/par[0])*par[2]/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')
                plt.errorbar(V*par[0]+par[2],mean_conductances*par[1],yerr=par[1]*std_conductance/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')

                plt.plot(V_data,res.CBT_model_G((V_data-par[2])/par[0])*par[1],label='1st order result for same parameters as the MC')
                plt.plot(V_data,f0(V_data,*par0),label='1st order result for optimal first order parameters: T={:.1f}mK'.format(1e3*par0[3]))
                plt.legend(loc=3)
                plt.ylabel('conductance [Si]')
                plt.xlabel('Bias voltage [V]')
                plt.tight_layout()
                try:
                    fig.savefig(res.filepath+'Chi_sq_plot.png')
                except FileNotFoundError:
                    os.mkdir(res.filepath)
                    fig.savefig(res.filepath+'Chi_sq_plot.png')
                    
                res.savedata()
                res.plotG()
                error_check=False
            except RuntimeError:
                print('The least square optimizer did not converge for these parameters')
                p_model[0]=p_model[0]+np.random.uniform(low=-1,high=1)*p_model[0]*1e-1
                p_model[1]=p_model[1]+np.random.uniform(low=-1,high=1)*p_model[1]*1e-1
                p_model[2]=p_model[2]+np.random.uniform(low=-1,high=1)*p_model[2]*1e-1
                print('Trying again with new parameters: '+str(p_model))
            if number_of_tries>1:
                print('number of tries:')
                print(number_of_tries)
                if number_of_tries>max_tries:
                    print('RuntimeError: leastq optimization failed for this run after  '+str(max_tries)+' number of tries')
                    error_check=False
        try:
            res.plotG(save=True)
        except Exception:
            pass #not important
        plt.close()
        ##########################################################################################
        # print('running the same simulation for q0=0')
        # lim=5.2*5.439*N
        # V=np.linspace(-lim,lim,points)
        # # number_of_concurrent=10
        # q0=0
        # offset_C=np.ones((N-1,))*0
        # dC=0
        # res=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
        #               parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC)
    
        # ####store main results###
        # mean_conductances=res.Gsm #mean conductance
        # std_conductance=res.Gstd #standard deviation of conductance
        # mean_currents=res.currentsm #mean currents
        # model=interp1d(V,mean_conductances,kind='linear',bounds_error=False,fill_value=(np.mean(mean_conductances[0:3]),np.mean(mean_conductances[-3::])))
        # wacky_sigma=interp1d(V,std_conductance,kind='linear',bounds_error=False,fill_value=(np.mean(std_conductance[0:3]),np.mean(std_conductance[-3::])))
        # def f(V,Ec,Gt,V0):
        #     return Gt*model((V-V0)/Ec)
        
        # p_model=[par0[0],par0[1],par0[2]]
        # error_check=True
        # number_of_tries=0
        # max_tries=5
        # while error_check==True:
        #     number_of_tries+=1
        #     try:
                
        #         par,cov=curve_fit(f,V_data,G_data,p0=p_model,sigma=V_data_std)
        #         print(par)
        #         G_MC=f(V_data,*par)
        #         chi_model=chi(G_data,G_MC,wacky_sigma(V_data/par[0])*par[2]/np.sqrt(number_of_concurrent))
        #         chi_0=chi(G_data,f0(V_data,*par0),np.mean(wacky_sigma(V_data/par[0])*par[2]/np.sqrt(number_of_concurrent)))
        #         chi_sq_0.append(chi_model)
        #         fig=plt.figure(figsize=(11,6))
        #         plt.errorbar(V_data,G_data,fmt='.',label='experimental data',yerr=G_data_std,xerr=V_data_std)

        #         plt.title('Best MC Fit parameters for u={:.2f}, q0={:.2f}e: '.format(u,q0)+' T={:.1f} mK'.format(1e3*par[0]/(u*kB))+'\n $G_T={:.1e}$'.format(par[1])+r' $\Omega^{-1}$'+' $E_c$={:.1e} $\mu$eV, $C_0$/C={:.2f}, <$\delta$ C>/C={:.2f}'.format(1e6*par[0],np.mean(offset_C),np.mean(dC)))
        #         # plt.errorbar(V_data,G_MC,yerr=wacky_sigma(V_data/par[0])*par[2]/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')
        #         plt.errorbar(V*par[0]+par[2],mean_conductances*par[1],yerr=par[2]*std_conductance/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')
        #         plt.ylabel('conductance [Si]')
        #         plt.xlabel('Bias voltage [V]')
        #         plt.tight_layout()
                
        #         plt.legend(loc=3)
        #         try:
        #             fig.savefig(res.filepath+'Chi_sq_plot1.png')
        #         except FileNotFoundError:
        #             os.mkdir(res.filepath)
        #             fig.savefig(res.filepath+'Chi_sq_plot1.png')
        #         fig=plt.figure(figsize=(11,6))
        #         plt.errorbar(V_data,G_data,fmt='.',label='experimental data',yerr=G_data_std,xerr=V_data_std)

        #         plt.title('Best MC Fit parameters for u={:.2f}, q0={:.2f}e: '.format(u,q0)+' T={:.1f} mK'.format(1e3*par[0]/(u*kB))+'\n $G_T={:.1e}$'.format(par[1])+r' $\Omega^{-1}$'+' $E_c$={:.1e} $\mu$eV'.format(1e6*par[0]))
        #         # plt.errorbar(V_data,G_MC,yerr=wacky_sigma(V_data/par[0])*par[2]/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')
        #         plt.errorbar(V*par[0]+par[2],mean_conductances*par[1],yerr=par[2]*std_conductance/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')

        #         plt.plot(V_data,res.CBT_model_G((V_data-par[2])/par[0])*par[1],label='1st order result for same parameters as the MC')
        #         plt.plot(V_data,f0(V_data,*par0),label='1st order result for optimal first order parameters: T={:.1e}mK'.format(1e3*par0[3]))
        #         plt.ylabel('conductance [Si]')
        #         plt.xlabel('Bias voltage [V]')
        #         plt.tight_layout()
        #         plt.grid()
        #         plt.legend(loc=3)
        #         try:
        #             fig.savefig(res.filepath+'Chi_sq_plot.png')
        #         except FileNotFoundError:
        #             os.mkdir(res.filepath)
        #             fig.savefig(res.filepath+'Chi_sq_plot.png')
                    
        #         res.savedata()
        #         res.plotG()
        #         error_check=False
        #     except RuntimeError:
        #         print('The least square optimizer did not converge for these parameters')
        #         p_model[0]=p_model[0]+np.random.uniform(low=-0.5,high=0.5)*p_model[0]*1e-1
        #         p_model[1]=p_model[1]+np.random.uniform(low=-0.5,high=0.5)*p_model[1]*1e-1
        #         p_model[2]=p_model[2]+np.random.uniform(low=-0.5,high=0.5)*p_model[2]*1e-1
        #         print('Trying again with new parameters: '+str(p_model))
        #     if number_of_tries>1:
        #         print('number of tries:')
        #         print(number_of_tries)
        #         if number_of_tries>max_tries:
        #             print('RuntimeError: leastq optimization failed for this run after  '+str(max_tries)+' number of tries')
        #             error_check=False
        # try:
        #     res.plotG(save=True)
        # except Exception:
        #     pass #not important
        plt.pause(0.05)
        plt.show()
        plt.close()
#%%
#folder_with_results='C:\\Users\\Elias Roos Hansen\\Documents\\Københavns uni\\qdev\\code\Runs with too much data'+'\\Results 2022-01-10 20.56.02.630781, sim time=1672.6sec\\'
path_to_data='C:\\Users\\Elias Roos Hansen\\Documents\\Københavns uni\\qdev\\code\Runs with too much data'
all_files = list()
dirname=list()
dirpaths=list()
models=list()
pars=list()
for (dirpath, dirnames, filenames) in os.walk(path_to_data):
    
    npz_files = list(filter(lambda x: x[-4:] == '.npz', filenames)) 
    print(npz_files)
    if len(npz_files) is not 0:
        all_files += npz_files
        dirname+=dirnames
        dirpaths+=[dirpath+'\\']
    
        fit_result=fit_carlo(V_data,G_data, filename=os.path.join(dirpath,npz_files[0]),plot=False,save_fig_folder=dirpath+'\\')
        if np.mean(fit_result.offset_C)>0:
            models.append((fit_result.model,fit_result.u))
            pars.append(fit_result.par)
    # plt.close()
meanGts=np.mean([p[1] for p in pars])
meanV0=np.mean([p[2] for p in pars])
meanEc=np.mean([p[0] for p in pars])
us=np.array([m[1] for m in models])
Ts=meanEc/(kB*us)

p0=[meanGts,us[0],0.1,meanV0,Ts[0]]
def super_model(V,Gt,u0,sig,V0,T):
    den=np.sum(np.array([np.exp(-(u-u0)**2/sig**2) for f,u in models]),axis=0)
    sup=Gt*np.sum(np.array([np.exp(-(u-u0)**2/sig**2)*f((V-V0)/(u*kB*T)) for f,u in models]),axis=0)/den
    return sup
bounds=([0,us[0],0.01,np.min(V_data),0.001],[1e-3,us[-1],1,np.max(V_data),1])
par,cov=curve_fit(super_model,V_data,G_data,p0=p0,bounds=bounds)
plt.figure()
plt.plot(V_data,G_data)
plt.plot(V_data,super_model(V_data,*par))
#%%

def ff(V_experiment,u):
    print(u)
    lim=2*5.439*N/u
    V=np.linspace(-lim,lim,points)
    res=carlo_CBT(V,1,u*kB,1,N=N,Nruns=9000,Ninterval=1000,Ntransient=10000,n_jobs=2,number_of_concurrent=8,
                  parallelization='external',q0=0,dV=5.439*N/(u*50),batchsize=10)


    ####store main results###
    mean_conductances=res.Gsm #mean conductance
    std_conductance=res.Gstd #standard deviation of conductance
    model=interp1d(V,mean_conductances,kind='quadratic')
    def f(V,Ec,Gt,V0):
        return Gt*model((V-V0)/Ec)
    p0=[4e-6,2.16e-5,0]
    par,cov=curve_fit(f,V_data,G_data,p0=p0,sigma=std_conductance)
    
    return