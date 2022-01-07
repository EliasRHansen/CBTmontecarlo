# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 12:16:13 2022

@author: Elias Roos Hansen
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import inv
from copy import copy
import random as random
from time import time,time_ns
from joblib import Parallel,delayed
from derivative import dxdt
import einops as eo
import os
from datetime import datetime
from numba import njit


np.seterr(all = 'raise')
kB=8.617*1e-5
e_SI=1.602*1e-19


def iterable(m):
    """
    

    Parameters
    ----------
    m : any type
        DESCRIPTION.

    Returns
    
    -------
    bool
        true if the object passed is iterable and false otherwise.

    """
    try:
        iter(m)
        return True
    except TypeError:
        return False

@njit
def pick_event2(x):
    i=x.shape[0]
    index=np.zeros((i),dtype='int32')
    for P in np.arange(i):
        index[P]=np.searchsorted(np.cumsum(x[P]), np.random.random(), side="right")
    return index
def split_voltages(V,dV):
    Us=V-dV
    Us=np.concatenate((Us,V+dV))
    return Us

class CBTmain: #just does the simulation, no further analysis
    
    def __init__(self,U,T,Ec,Gt,N,Nruns,Ntransient,number_of_concurrent,Ninterval,skip_transient,parallelization='external',
                 n0=None,second_order_C=None,dtype='float64',offset_C=None,dC=0,n_jobs=2,batchsize=1,q0=0):

        if n0 is None:
            n0=np.array([0]*(N-1))+q0
        else:
            self.n0=n0+q0
        if second_order_C is None:
            second_order_C=np.zeros((N,),dtype=dtype)
            self.second_order_C=second_order_C#units of e/Ec=160[fmF]/Ec[microeV]
        else:
            self.second_order_C=second_order_C.astype(dtype)#units of e/Ec=160[fmF]/Ec[microeV]
        if offset_C is None:
            offset_C=np.zeros((N-1,),dtype=dtype)
            self.offset_C=offset_C
        else:
            self.offset_C=offset_C.astype(dtype)#units of e/Ec=160[fmF]/Ec[microeV]
            
        self.Ec=Ec #units of eV
        self.N=N
        self.n0=n0.astype(dtype) #units of number of electrons
        self.Cs=np.ones((N,),dtype=dtype)+dC #units of e/Ec=160 [fmF]/Ec[microeV]
        self.U=U #units of eV
        self.Nruns=Nruns
        self.Ntransient=Ntransient
        self.Ninterval=Ninterval
        self.skip_transient=skip_transient
        self.parallelization=parallelization
        self.batchsize=batchsize
        self.q0=q0
        if iterable(U):
            self.number_of_Us=len(U)
            
            
        else:
            if self.parallelization !='non':
                print('voltage is a scalar; setting parallelization to non, since parallelization is not possible')
                self.parallelization='non'
            self.number_of_Us=1
        # self.neff=self.neff_f(self.n0)
        self.T=T
        self.kBT=kB*T #units of eV
        self.u=self.Ec/(self.kBT)
        self.Gt=Gt
        self.dtype=dtype
        self.factor_SI=e_SI/(self.N*self.Ec*self.Gt)
        
        d1=np.concatenate((self.Cs,np.zeros((1,))))
        dm1=np.concatenate((self.Cs,np.zeros((1,))))
        d2=np.concatenate((self.second_order_C[0:-1],np.zeros((2,))))
        dm2=np.concatenate((self.second_order_C[1::],np.zeros((2,))))
        dm1=np.roll(dm1,-1)
        dm2=np.roll(dm2,-1)
        d0=np.roll(dm2,2)+np.roll(dm1,1)+np.roll(d1,-1)+np.roll(d2,-2)+np.concatenate((self.offset_C,np.zeros((2,))))
        data=np.array([-dm2,-dm1,d0,-d1,-d2])
        data=data[:,0:-2]
        offsets=np.array([-2,-1,0,1,2])
        self.C=sparse.dia_matrix((data,offsets),shape=(N-1,N-1),dtype=dtype)
        self.Cinv=inv(sparse.csc_matrix(self.C)).toarray()
        dataM=np.array([[-1]*self.N,[1]*self.N])
        offsetsM=np.array([0,1])
        self.M=sparse.dia_matrix((dataM,offsetsM),shape=(self.N-1,self.N),dtype=dtype).toarray()
        self.MM=np.concatenate((self.M,-self.M),axis=1)
        transient=int(Nruns/Ninterval)
        self.now=str(datetime.now()).replace(':','.')
        print('running '+str(number_of_concurrent)+'simulations initiated after a transient period of'+str(Ntransient)+'steps. This is done for '+str(self.number_of_Us)+' voltages, that are run in "internal" batches of size '+str(batchsize)+'.')
        print('The total number of simulations (that gives rise to a datapoint for current) is: '+str(number_of_concurrent*self.number_of_Us))
        a=time()
        if (self.parallelization=='internal'):
            if batchsize>1:
                voltages=np.array(self.U)
                voltbatch=[]
                for j in np.arange(int(np.ceil(self.number_of_Us/batchsize))):

                    voltbatch.append(voltages[j*batchsize:batchsize*(j+1)])
                    if len(voltbatch[-1])<batchsize:
                        print('number of Us is not divisible into batches of size '+str(batchsize)+'. This is handled by having the last batch size being: '+str(len(voltbatch[-1])))
                print(voltbatch)
                self.dQps=[]
                self.dtps=[]
                for V in voltbatch:
                    self.U=V
                    self.number_of_Us=len(V)
                    self.update_number_of_concurrent(number_of_concurrent)
                    self.__call__(number_of_steps=Nruns,transient=transient,print_every=Ninterval,number_of_concurrent=number_of_concurrent,skip_transient=skip_transient)
                    self.dQps.append(self.dQp)
                    self.dtps.append(self.dtp)
                self.voltbatch=voltbatch
            else:
                print('batchsize is one, but parallization is True, so all voltages a run. To run one at a time, use paralization="non"')
                self.update_number_of_concurrent(number_of_concurrent)
                self.__call__(number_of_steps=Nruns,transient=transient,print_every=Ninterval,number_of_concurrent=number_of_concurrent,skip_transient=skip_transient)
            
        elif (self.parallelization=='non'):
            if batchsize>1:
                print('parallization set to non; batchsize doesnt have any effect')
            if self.number_of_Us==1:
                self.update_number_of_concurrent(number_of_concurrent)
                self.__call__(number_of_steps=Nruns,transient=transient,print_every=Ninterval,number_of_concurrent=number_of_concurrent,skip_transient=skip_transient)
            else:
                voltages=np.array(self.U)
                self.number_of_Us=1
                
                self.dQps=[]
                self.dtps=[]
                for V in voltages:
                    self.U=V
                    self.update_number_of_concurrent(number_of_concurrent)
                    self.__call__(number_of_steps=Nruns,transient=transient,print_every=Ninterval,number_of_concurrent=number_of_concurrent,skip_transient=skip_transient)
                    self.dQps.append(self.dQp)
                    self.dtps.append(self.dtp)
            
        elif (self.parallelization=='external'):
            self.number_of_concurrent=number_of_concurrent #otherwise it never gets to be an attribute since __call__ is called inside joblib
            voltages=np.array(self.U)
            if batchsize>1:
                voltbatch=[]
                for j in np.arange(int(np.ceil(self.number_of_Us/batchsize))):

                    voltbatch.append(voltages[j*batchsize:batchsize*(j+1)])
                    if len(voltbatch[-1])<batchsize:
                        print('number of Us is not divisible into batches of size '+str(batchsize)+'. This is handled by having the last batch size being: '+str(len(voltbatch[-1])))
                print('the total number of tasks is ' +str(voltbatch))
                self.voltbatch=voltbatch
            
            def f_scalar(V):
                self.U=V
                self.number_of_Us=1

                self.update_number_of_concurrent(number_of_concurrent)
                self.__call__(number_of_steps=Nruns,transient=transient,print_every=Ninterval,number_of_concurrent=number_of_concurrent,skip_transient=skip_transient)
                return self.dQp,self.dtp
            def f_vector(vbatch):
                self.U=vbatch
                self.number_of_Us=len(vbatch)

                self.update_number_of_concurrent(number_of_concurrent)
                self.__call__(number_of_steps=Nruns,transient=transient,print_every=Ninterval,number_of_concurrent=number_of_concurrent,skip_transient=skip_transient)
                return self.dQp,self.dtp
            if batchsize==1:
                print('the total number of tasks is ' +str(voltages))
                self.Is=Parallel(n_jobs=n_jobs,verbose=50)(delayed(f_scalar)(U) for U in voltages)
            else:
                self.Is=Parallel(n_jobs=n_jobs,verbose=50)(delayed(f_vector)(U) for U in voltbatch)
        try:
            self.U=voltages
            self.number_of_Us=len(voltages)
        except NameError:
            #'voltages' was not defined, so U wasnt redefined and it is unecessary to reset it.
            pass
        b=time()
        self.simulation_time=b-a
        print('simulation time: '+str(self.simulation_time))

    def update_number_of_concurrent(self,number_of_concurrent):
        self.number_of_concurrent=number_of_concurrent
        self.MMM_withoutU=np.tile(self.MM,(1,number_of_concurrent))
        self.MMM=np.tile(self.MM,(1,number_of_concurrent*self.number_of_Us))
        self.B_withoutU=self.Cinv@self.MMM_withoutU
        self.BB=self.Cinv@self.MM
        self.BB=np.array(self.BB,dtype=self.dtype)
        if iterable(self.U)==False:
            C=np.einsum('ij,ij->j',self.MMM_withoutU,self.B_withoutU)/2
            
            self.dE0=C+self.U/(self.Ec)*(self.Cs[0]*self.B_withoutU[0,:]-self.Cs[-1]*self.B_withoutU[-1,:]+self.second_order_C[1]*self.B_withoutU[1,:]-self.second_order_C[-1]*self.B_withoutU[-2,:])
            self.boundary_works=self.U/(2*self.Ec)
        else:

            self.B=self.Cinv@self.MMM
            C=np.einsum('ij,ij->j',self.MMM,self.B)/2
            bound=np.kron(self.U/(self.Ec),(self.Cs[0]*self.B_withoutU[0,:]-self.Cs[-1]*self.B_withoutU[-1,:]+self.second_order_C[1]*self.B_withoutU[1,:]-self.second_order_C[-1]*self.B_withoutU[-2,:]))
            self.boundary_works=self.U.repeat(number_of_concurrent)/(2*self.Ec)
            self.dE0=C+bound
        # else:
        #     print(str(self.parallelization)+' is not implemented as a value for the parameter parallelization')
            
    def dE_f(self,nn):
        v=(nn.T@self.BB).flatten()
        # print(v.dtype)
        E=v+self.dE0#units of Ec
        E[::2*self.N]=E[::2*self.N]+self.boundary_works
        E[self.N-1::2*self.N]=E[self.N-1::2*self.N]+self.boundary_works
        E[self.N::2*self.N]=E[self.N::2*self.N]-self.boundary_works
        E[2*self.N-1::2*self.N]=E[2*self.N-1::2*self.N]-self.boundary_works

        return E
    def update_transition_rate(self,n1,update_gammas=True):

        limit1=1e-12
        limit2=1e12

        dE=-self.dE_f(n1)
        try:
            Gamma=np.array(self.gammas)#np.zeros_like(dE)
        except AttributeError:
            Gamma=np.zeros_like(dE)
            
        if dE.ndim==1:
            # try:
            #     Gamma=-dE/np.expm1(-dE*self.u)#(1-np.exp(-dE*self.u))
            # except FloatingPointError:
            c1=-dE*self.u>np.log(limit1)
            c2=-dE*self.u<np.log(limit2)
            c5=-dE*self.u<=np.log(limit1)
            c6=-dE*self.u>=np.log(limit2)
            dE1=dE[(c1) & (c2)]
            try:
                Gamma[(c1) & (c2)]=-dE1/np.expm1(-dE1*self.u)#(1-np.exp(-dE1*self.u))
            except FloatingPointError:
                print('a floating point error occurred for dE[..]='+str(dE1))
                c3=-dE*self.u<0
                c4=-dE*self.u>0
                dE3=dE[(c1) & (c3)]
                dE4=dE[(c4) & (c2)]
                Gamma[(c1) & (c3)]=dE3/(1-np.exp(-dE3*self.u))
                Gamma[(c4) & (c2)]=dE4/(1-np.exp(-dE4*self.u))
                Gamma[dE*self.u==0.]=1/(self.u)
            Gamma[c5]=dE[c5]
            try:
                dE2=dE[c6]
                Gamma[c6]=-dE2*np.exp(dE2*self.u)
            except FloatingPointError:
                Gamma[c6]=0
            # print('updating transition rates')
            Gamma=Gamma
            if update_gammas:
                self.gammas=Gamma
                self.sumgammas=sum(self.gammas)
            return Gamma

        else:
            print('something is wrong with the dimensions of energy difference')
    def P(self,n,update=False):
        """
        

        Parameters
        ----------
        n : current charge configuration
            DESCRIPTION.

        Returns
        -------
        array of 2N transition probabilities where the first N represents a transition from Right to left and the second N represents transitions from left to right.
            DESCRIPTION.

        """
        try:
            
            p=self.gammas2
        except Exception:
            p=self.update_transition_rate(n).reshape(self.number_of_concurrent*self.number_of_Us,2*self.N)
            
        self.p=np.einsum('ij,i->ij',p,1/self.Gamsum)

        return self.p

    def dt_f(self,n):

        self.dts=self.factor_SI/self.Gamsum
        return self.dts
    def dQ_f(self,n):

        self.dQ=-(e_SI/self.N)*self.Gamdif/self.Gamsum
        return self.dQ
            
    def multistep(self,store_data=False):

            try:
                neff=self.ns

                self.gammas2=self.update_transition_rate(neff).reshape(self.number_of_concurrent*self.number_of_Us,2*self.N)
                self.Gamsum=np.sum(self.gammas2,axis=1)
                self.Gamdif=np.sum(self.gammas2[:,0:self.N]-self.gammas2[:,self.N::],axis=1)
                self.P(neff)
                if store_data:
                    self.dtp.append(self.dt_f(neff))
                    self.dQp.append(self.dQ_f(neff))

                self.indices=pick_event2(self.p)#self.pick_event(neff,1)
                self.indices=list(self.indices)
                n_new=neff+self.MM[:,self.indices]
                self.ns=n_new
                
            except FloatingPointError:
                print('FloatingPointError Occurred; trying to redo step. This may occur when the sum of the transition rates is very small. In this case, the sums are: '+str(self.Gamsum)+". However, I think the bug causing this is gone now.")
                print(np.sum(self.p,axis=1))
                self.indices=pick_event2(self.p)#self.pick_event(neff,1)
                self.indices=list(self.indices)
                n_new=neff+self.MM[:,self.indices]#[:,:,0]
                self.ns=n_new
                self.multistep()
    
    def __call__(self,number_of_steps=1,transient=0,print_every=None,number_of_concurrent=None,skip_transient=True):
        if number_of_concurrent is None :
            number_of_concurrent=copy(self.number_of_concurrent)
        try:
            print('resetting attributes')
            del self.gammas

        except Exception:
            print('attribute gammmas does not exist')
        try:

            del self.gammas2
        except Exception:
            print('attribute gammas2 does not exist')
        try:
            del self.p

        except Exception:
            print('attribute p does not exist')
        self.dtp=[]
        self.dQp=[]

        if skip_transient:
            print('n0 is being moved forward through the transient window')

            self.update_number_of_concurrent(1)
            
            self.ns=np.array([self.n0]*self.number_of_Us).T
            print('initiating multistep for the transient window for '+str(self.number_of_concurrent)+' charge configurations to move through the transient regime')
            for j in np.arange(transient*print_every):
                self.multistep(store_data=False)
                
            # self.n0=np.array(self.n)
            
            del self.gammas
            del self.p
        else:
            self.ns=np.array([self.n0]*self.number_of_Us).T
        self.update_number_of_concurrent(number_of_concurrent)

        self.ns=self.ns.repeat(self.number_of_concurrent,axis=1)
        print('initiating multistep for the transient window for '+str(self.number_of_concurrent*self.number_of_Us)+' charge configurations')
        # print('initial charge configuration (columns):')
        # print(self.n0)
        if print_every is None:
            print_every=int(number_of_steps/100+1)
        for i in np.arange(number_of_steps):
            self.progress_index=i
            if print_every != 0:
                if i%print_every==print_every-1:
                    print('{:.1f}'.format(i*100/number_of_steps)+' pct.')
        
                    self.multistep(store_data=True)
                else:
                    self.multistep(store_data=False)
            else:
                self.multistep(store_data=False)
        print('done')
        



class CBT_data_analysis:
    def __init__(self,CBTmain_instance,transient=None):
        CBT=CBTmain_instance #shorthandname
        self.raw_data=CBT
        self.now=CBT.now
        self.simulation_time=CBT.simulation_time
        if transient is None:
            if CBT.skip_transient:
                transient=0
        if CBT.parallelization=='external':
            if CBT.batchsize==1:
                self.dQ=np.array([I[0] for I in CBT.Is]) #shape->(voltages,along time dimensio, parallel to time dimension)
                self.dt=np.array([I[1] for I in CBT.Is]) #shape->(voltages,along time dimensio, parallel to time dimension)
            else:
                voltbatch=CBT.voltbatch
                batchlengths=[len(v) for v in voltbatch]
                nc=CBT.number_of_concurrent
                Us=CBT.U
                number_of_time_steps_with_data=len(CBT.Is[0][0])
                dQ=np.empty((len(Us),number_of_time_steps_with_data,nc))
                dt=np.empty((len(Us),number_of_time_steps_with_data,nc))
                for j in np.arange(len(CBT.Is)):
                    i1,i2=CBT.Is[j]
                    dQ1=np.array(i1).reshape(number_of_time_steps_with_data,batchlengths[j],nc)
                    dt1=np.array(i2).reshape(number_of_time_steps_with_data,batchlengths[j],nc)
                    dQ1=np.array(dQ1).swapaxes(1,0)
                    dt1=np.array(dt1).swapaxes(1,0)
                    
                    dQ[sum(batchlengths[0:j]):sum(batchlengths[0:j+1]),...]=dQ1
                    dt[sum(batchlengths[0:j]):sum(batchlengths[0:j+1]),...]=dt1
                self.dQ=dQ
                self.dt=dt

        elif CBT.parallelization=='internal':
            if CBT.batchsize==1:
                Us=CBT.U
                number_of_time_steps_with_data=len(CBT.dQp)
                nc=CBT.number_of_concurrent
                
                self.dQ=np.array(CBT.dQp).reshape(number_of_time_steps_with_data,len(Us),nc).swapaxes(1,0)
                self.dt=np.array(CBT.dtp).reshape(number_of_time_steps_with_data,len(Us),nc).swapaxes(1,0)
            else:
                voltbatch=CBT.voltbatch
                batchlengths=[len(v) for v in voltbatch]
                nc=CBT.number_of_concurrent
                Us=CBT.U
                number_of_time_steps_with_data=len(CBT.dQp)
                dQ=np.empty((len(Us),number_of_time_steps_with_data,nc))
                dt=np.empty((len(Us),number_of_time_steps_with_data,nc))
                for j in np.arange(len(CBT.dQp)):
                    i1=CBT.dQp[j]
                    i2=CBT.dtp[j]
                    dQ1=np.array(i1).reshape(number_of_time_steps_with_data,batchlengths[j],nc)
                    dt1=np.array(i2).reshape(number_of_time_steps_with_data,batchlengths[j],nc)
                    dQ1=np.array(dQ1).swapaxes(1,0)
                    dt1=np.array(dt1).swapaxes(1,0)
                    
                    dQ[sum(batchlengths[0:j]):sum(batchlengths[0:j+1]),...]=dQ1
                    dt[sum(batchlengths[0:j]):sum(batchlengths[0:j+1]),...]=dt1
                self.dQ=dQ
                self.dt=dt
        elif CBT.parallelization=='non':
                Us=CBT.U
                number_of_time_steps_with_data=len(CBT.dQp[0])
                dQ=np.empty((len(Us),number_of_time_steps_with_data,nc))
                dt=np.empty((len(Us),number_of_time_steps_with_data,nc))
                for j in np.arange(len(CBT.dQp)):
                    i1=CBT.dQp[j]
                    i2=CBT.dtp[j]
                    dQ1=np.array(i1).reshape(number_of_time_steps_with_data,1,nc)
                    dt1=np.array(i2).reshape(number_of_time_steps_with_data,1,nc)
                    dQ1=np.array(dQ1).swapaxes(1,0)
                    dt1=np.array(dt1).swapaxes(1,0)
                    
                    dQ[j:j+1,...]=dQ1
                    dt[j:j+1,...]=dt1
                self.dQ=dQ
                self.dt=dt
        self.currents=np.sum(np.array(self.dQ)[:,transient::,:],axis=1)/np.sum(np.array(self.dt)[:,transient::,:],axis=1)
        self.currentsm=np.mean(self.currents,axis=1)
        self.currentsstd=np.std(self.currents,axis=1)
        self.points=int(len(self.currentsm)/2)
        points=self.points
        Us=CBT.U
        
        self.Gs=(self.currents[points:2*points,:]-self.currents[0:points,:])/np.array([(Us[points:2*points]-Us[0:points])]).repeat(len(self.currents[0,:]),axis=0).T
        self.Gsm=np.sum(self.Gs,axis=1)/len(self.Gs[0,:])
        self.Gstd=np.std(self.Gs,axis=1)
        self.dV=Us[points:2*points]-Us[0:points]
    def CBT_model_g(self,x):
        return (x*np.sinh(x)-4*np.sinh(x/2)**2)/(8*np.sinh(x/2)**4)

    def CBT_model_G(self,V):
        return self.raw_data.Gt*(1-self.raw_data.Ec*self.CBT_model_g(V/(self.raw_data.N*kB*self.raw_data.T))/(kB*self.raw_data.T))
    def plotG(self,save=False):

        points=self.points
        fig,ax=plt.subplots(figsize=(9,6))
        plt.title('MC for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={:.1e}e, steps between samples={}, \n steps/(run'.format(self.raw_data.N,self.raw_data.T*1e3,self.raw_data.Ec*1e6,self.raw_data.Gt*1e6,self.raw_data.q0,
                                                                                                                                                            self.raw_data.Ninterval)+r'$\times$'+'datapoint)={}, runs/datapoint={}, transient interval={}'.format(self.raw_data.Nruns,self.raw_data.number_of_concurrent,self.raw_data.Ntransient))
        
        Us=self.raw_data.U
        Vs=(Us[points:2*points]+Us[0:points])/2
        for i in np.arange(len(self.Gs[0,:])):
            plt.plot(Vs,self.Gs[:,i],'.',color=[(i+1)/len(self.Gs[0,:]),0,0])
        ax.plot(np.linspace(min(Us),max(Us),1000),self.CBT_model_G(np.linspace(min(Us),max(Us),1000)),label='First order analytic result')
        ax.set_xlabel('Bias voltage [V]')
        ax.set_ylabel(r'Differential Conductance [Si], ($\Delta$ I/$\Delta$ V, $\Delta$ V={:.1e} eV)'.format(np.mean(self.dV[0])))
        ax.legend()
        plt.grid()
        plt.tight_layout()
        fig2,ax2=plt.subplots(figsize=(9,6))

        plt.title('Results for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={:.1e}e'.format(self.raw_data.N,self.raw_data.T*1e3,self.raw_data.Ec*1e6,self.raw_data.Gt*1e6,self.raw_data.q0))
        ax2.errorbar(Vs,self.Gsm,yerr=self.Gstd/np.sqrt(len(self.Gs[0,:])),fmt='.',label='Monte Carlo simulation results (mean)')
        ax2.plot(np.linspace(min(Us),max(Us),1000),self.CBT_model_G(np.linspace(min(Us),max(Us),1000)),label='first order analytic result')
        ax2.set_xlabel('bias voltage [V]')
        ax2.set_ylabel(r'Differential Conductance [Si], ($\Delta$ I/$\Delta$ V, $\Delta$ V={:.1e} eV)'.format(np.mean(self.dV)))
        ax2.legend()
        plt.grid()
        plt.tight_layout()
        
        if save==True:
        
            filepath=os.getcwd()
            try:
                fig.savefig(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)+'Conductance1.png')
                fig2.savefig(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)+'Conductance2.png')
                print('saving figures in folder: '+filepath)
            except Exception:
                print('saving figures in folder: '+filepath)
                os.mkdir(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time))
                fig.savefig(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)+'Conductance1.png')
                fig2.savefig(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)+'Conductance2.png')
    def plotI(self,save=False):
        fig,ax=plt.subplots(figsize=(9,6))
        plt.title('MC for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={:.1e}e, steps between samples={}, \n steps/(run'.format(self.raw_data.N,self.raw_data.T*1e3,self.raw_data.Ec*1e6,self.raw_data.Gt*1e6,self.raw_data.q0,
                                                                                                                                                            self.raw_data.Ninterval)+r'$\times$'+'datapoint)={}, runs/datapoint={}, transient interval={}'.format(self.raw_data.Nruns,self.raw_data.number_of_concurrent,self.raw_data.Ntransient))
        
        Us=self.raw_data.U

        for i in np.arange(len(self.currents[0,:])):
            plt.plot(Us,self.currents[:,i],'.',color=[(i+1)/len(self.currents[0,:]),0,0])
        # ax.plot(np.linspace(min(Us),max(Us),1000),self.CBT_model_G(np.linspace(min(Us),max(Us),1000)),label='First order analytic result')
        ax.set_xlabel('Bias voltage [V]')
        ax.set_ylabel('Current [A]')
        plt.grid()
        plt.tight_layout()
        fig2,ax2=plt.subplots(figsize=(9,6))

        plt.title('Results for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={:.1e}e'.format(self.raw_data.N,self.raw_data.T*1e3,self.raw_data.Ec*1e6,self.raw_data.Gt*1e6,self.raw_data.q0))
        ax2.errorbar(Us,self.currentsm,yerr=self.currentsstd,fmt='.',label='Monte Carlo simulation results (mean)')
        ax2.set_xlabel('bias voltage [V]')
        ax2.set_ylabel('Current [A]')
        ax2.legend()
        plt.grid()
        plt.tight_layout()
        if save==True:
        
            filepath=os.getcwd()
            try:
                fig.savefig(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)+'Current1.png')
                fig2.savefig(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)+'Current2.png')
                print('saving figures in folder: '+filepath)
            except Exception:
                print('saving figures in folder: '+filepath)
                os.mkdir(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time))
                fig.savefig(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)+'Current1.png')
                fig2.savefig(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)+'Current2.png')
        # self.CBTmain_instance=CBTmain_instance

def carlo_CBT(U,T,Ec,Gt,N=100,Nruns=5000,Ntransient=5000,number_of_concurrent=5,Ninterval=1000,skip_transient=True,parallelization='external',
             n0=None,second_order_C=None,dtype='float64',offset_C=None,dC=0,n_jobs=2,batchsize=1,q0=0,split_voltage=True,dV=None,
             make_plots=False,save_plots=True,output='full'):
    
    outputs=['full','G_mean, G_std','I_mean, I_std','G','I']
    if output not in outputs:
        print('output parameter must be one of '+str(outputs))
        raise Exception('wrong output parameter')
    FWHM=5.439*kB*T*N
    if dV is None:
        dV=FWHM/50
    if dV>FWHM/5:
        print('WARNING: dV is VERY HIGH!!, the conductance will a large systematic error!')
    if number_of_concurrent==1:
        print('warning: if number_of_concurrent is 1, standard deviations cannot be calculated by the current method')
    if split_voltage:
        Vs=split_voltages(U,dV)
    else:
        print('warning: the conductance will be completely wrong if split_voltage is not true, unless V is specified such that every second value in ascending sense is ordered on the left/right side of the array')
        Vs=V
    raw_result=CBTmain(Vs,T,Ec,Gt,N,Nruns=Nruns,
                       Ntransient=Ntransient,
                       number_of_concurrent=number_of_concurrent,
                       Ninterval=Ninterval,
                       skip_transient=skip_transient,
                       parallelization=parallelization,
                       n0=n0,
                       second_order_C=second_order_C,
                       dtype=dtype,
                       offset_C=offset_C,
                       dC=dC,
                       n_jobs=n_jobs,
                       batchsize=batchsize,
                       q0=q0)
    result=CBT_data_analysis(raw_result)
    if dV>FWHM/5:
        print('WARNING: dV is VERY HIGH!!, the conductance will a large systematic error!')
    if make_plots:
        result.plotG(save=save_plots)
        result.plotI(save=save_plots)
    if output=='full':
        
        return result
    elif output=='G_mean':
        return result.Gsm
    elif output=='G_mean, G_std':
        return result.Gsm,result.Gstd
    elif output=='I_mean, I_std':
        return result.currentsm,result.currentsstd
    elif output=='G':
        return result.G
    elif output=='I':
        return result.currents
    
if __name__=='__main__': #runs only if the file is being run explicitly
    
    ###################################################For testing########################################
    
    N=100 #Number of islands
    Ec=4e-6 #Charging energy in units of eV
    Gt=2e-5 #Large voltage asymptotic conductance (affects noly the scaling of the result)
    T=0.01 #Temperature in Kelvin
    FWHM=5.439*kB*T*N #Full width half max according to the first order model

    points=11 #number of voltages to run the simulation for
    lim=3*FWHM 
    V=np.linspace(-lim,lim,points)
    
    
    ####Run main simulation####
    print('runing example')
    res=carlo_CBT(V,T,Ec,Gt,N=100,Nruns=6000,Ninterval=1000,Ntransient=10000,n_jobs=4,parallelization='external')
    print('finished running example')
    
    ####store main results###
    print('making plots')
    mean_conductances=res.Gsm #mean conductance
    std_conductance=res.Gstd #standard deviation of conductance
    mean_currents=res.currentsm #mean currents
    res.plotG(save=True)
    res.plotI(save=True)
    
    