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
from time import time
from joblib import Parallel,delayed
from derivative import dxdt
import einops as eo
import os
from datetime import datetime
from numba import njit
np.seterr(all = 'raise')
#########Inpu
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

class CBTmain: #just does the simulation, no further analysis
    
    def __init__(self,U,T,Ec,Gt,N,Nruns,Ntransient,number_of_concurrent,Ninterval,skip_transient,parallelization='external',
                 n0=None,second_order_C=None,dtype='float64',offset_C=None,dC=0,n_jobs=4):

        if n0 is None:
            n0=np.array([0]*(N-1))
        else:
            self.n0=n0
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
        
        if iterable(U):
            self.number_of_Us=len(U)
            if self.parallelization=='external':
                print('ERROR: if parallelization is external, voltages (U) must be scalars, not array-like. Setting parallelization to internal!')
                self.parallelization='internal'
        else:
            self.number_of_Us=1
        # self.neff=self.neff_f(self.n0)
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
        a=time()
        if (parallelization=='internal'):
            self.update_number_of_concurrent(number_of_concurrent)
            self.__call__(number_of_steps=Nruns,transient=transient,print_every=Ninterval,number_of_concurrent=number_of_concurrent,skip_transient=skip_transient)
        elif (parallelization=='non'):
            
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
                    
        elif (parallelization=='external'):
            voltages=np.array(self.U)
            self.number_of_Us=1
            def f(V):
                self.U=V
                self.update_number_of_concurrent(number_of_concurrent)
                self.__call__(number_of_steps=Nruns,transient=transient,print_every=Ninterval,number_of_concurrent=number_of_concurrent,skip_transient=skip_transient)
                
                return self.dQp,self.dtp
            self.Is=Parallel(n_jobs=n_jobs,verbose=50)(delayed(f)(U) for U in voltages)
        b=time()
        self.simulation_time=b-a

    def update_number_of_concurrent(self,number_of_concurrent):
        self.number_of_concurrent=number_of_concurrent
        self.MMM_withoutU=np.tile(self.MM,(1,number_of_concurrent))
        self.MMM=np.tile(self.MM,(1,number_of_concurrent*self.number_of_Us))
        self.B_withoutU=self.Cinv@self.MMM_withoutU
        self.BB=self.Cinv@self.MM
        
        if self.parallization=='external' or self.parallization=='non':
            C=np.einsum('ij,ij->j',self.MMM_withoutU,self.B_withoutU)/2
            
            self.dE0=C+self.U/(self.Ec)*(self.Cs[0]*self.B_withoutU[0,:]-self.Cs[-1]*self.B_withoutU[-1,:]+self.second_order_C[1]*self.B_withoutU[1,:]-self.second_order_C[-1]*self.B_withoutU[-2,:])
            self.boundary_works=self.U/(2*self.Ec)
        elif self.parallelization=='internal':

            self.B=self.Cinv@self.MMM
            C=np.einsum('ij,ij->j',self.MMM,self.B)/2
            bound=np.kron(self.U/(self.Ec),(self.Cs[0]*self.B_withoutU[0,:]-self.Cs[-1]*self.B_withoutU[-1,:]+self.second_order_C[1]*self.B_withoutU[1,:]-self.second_order_C[-1]*self.B_withoutU[-2,:]))
            self.boundary_works=self.U.repeat(number_of_concurrent)/(2*self.Ec)
            self.dE0=C+bound
        else:
            print(str(self.parallelization)+' is not implemented as a value for the parameter parallelization')
            
    def dE_f(self,nn):
        v=(nn.T@self.BB).flatten()
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
            try:
                Gamma=dE/(1-np.exp(-dE*self.u))
            except FloatingPointError:
                c1=-dE*self.u>np.log(limit1)
                c2=-dE*self.u<np.log(limit2)
                c5=-dE*self.u<=np.log(limit1)
                c6=-dE*self.u>=np.log(limit2)
                dE1=dE[(c1) & (c2)]
                try:
                    Gamma[(c1) & (c2)]=dE1/(1-np.exp(-dE1*self.u))
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
            # Gamma=Gamma
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
            
            p=self.gammas
        except Exception:
            p=self.update_transition_rate(n)
        try:
            Norm=self.sumgammas
        except Exception:
            Norm=np.sum(p)

        if self.number_of_concurrent*self.number_of_Us==1:
                
            try:
                self.p=p/Norm
            except FloatingPointError:
                print('probability not normalized due to floting point error')
            return self.p
        else:
            
            self.p=np.einsum('ij,i->ij',self.gammas2,1/self.Gamsum)

            return self.p

    def dt_f(self,n):

        if self.number_of_concurrent*self.number_of_Us==1:
            try:
                # self.dts=factor_SI/(self.gammas[0:self.N]+self.gammas[self.N::])
                self.dts=self.factor_SI/self.sumgammas
                return self.dts
                # return sum(self.dts)
            except Exception:
                self.update_transition_rate(n)
                # self.dts=factor_SI/(self.gammas[0:self.N]+self.gammas[self.N::])
                return self.dts
                self.dts=self.factor_SI/self.sumgammas
                # return sum(self.dts)
        else:
                self.dts=self.factor_SI/self.Gamsum
                return self.dts
    def dQ_f(self,n):
        if self.number_of_concurrent==1:
            try:
                self.dQ=-(e_SI/self.N)*sum(self.gammas[0:self.N]-self.gammas[self.N::])/self.sumgammas
                return self.dQ
            except Exception:
                self.update_transition_rate(self.Q(n),self.Q0(n))
                self.dQ=-(e_SI/self.N)*sum(self.gammas[0:self.N]-self.gammas[self.N::])/self.sumgammas
                return self.dQ
        else:
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
            print('initiating multistep for the transient window for '+str(self.Number_of_concurrent)+' charge configurations')
            for j in np.arange(transient*print_every):
                self.multistep(store_data=False)
                
            # self.n0=np.array(self.n)
            
            del self.gammas
            del self.p
        else:
            self.ns=np.array([self.n0]*self.number_of_Us).T
        self.update_number_of_concurrent(number_of_concurrent)
        print(self.ns)
        self.ns=self.ns.repeat(self.number_of_concurrent,axis=1)
        print('initiating multistep for the transient window for '+str(self.Number_of_concurrent*self.number_of_concurrent)+' charge configurations')
        print('initial charge configurations (columns):')
        print(self.ns)
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



class CBT_results:
    def __init__(self,CBTmain_instance):
        CBT=CBTmain_instance #shorthandname
        if CBT.parallelization=='external':
            pass
        elif CBT.parallelization=='internal':
            pass
        elif CBT.parallelization=='non':
            pass
        self.CBTmain_instance=CBTmain_instance
        
