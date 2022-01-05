# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:47:45 2022

@author: Elias Roos Hansen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 14:09:59 2021

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
# @guvectorize(["int32[:,:](float64[:,:])"],
#              "(m,n)->(m,n)")
# @njit
# def pick_event2(x):
    
#     i=x.shape[0]
#     j=x.shape[1]
#     index=np.zeros((i))
#     for P in np.arange(i):
#         index[P]=np.random.choice(j,p=x[P])
#     return index
@njit
def pick_event2(x):
    i=x.shape[0]
    index=np.zeros((i),dtype='int32')
    for P in np.arange(i):
        index[P]=np.searchsorted(np.cumsum(x[P]), np.random.random(), side="right")
    return index

class CBTmontecarlo:
    
    def __init__(self,N,T,Ec,Gt,offset_C=None,second_order_C=None,n0=None,gi=None,U=np.array([0]),dtype='float64',number_of_concurrent=1,dC=None):
        if dC is None:
            dC=0
            self.dC=dC
        else:
            self.dC=dC
        if gi is None:
            gi=1
            self.gi=gi
        else:
            self.gi=gi
        if n0 is None:
            n0=np.array([0]*(N-1))
        # elif n0=='random':
        #     n0=np.array(random.choices(np.arange(11)-5,k=N-1,weights=np.exp(-0.2*(np.arange(11)-5)**2)))
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
        self.U=U #units of eV
        self.number_of_Us=len(U)
        
        self.Ec=Ec #units of eV
        self.N=N
        self.n0=n0.astype(dtype) #units of number of electrons
        self.n=n0.astype(dtype)#np.array([n0.astype(dtype)]*self.ofof_Us).T
        
        self.Cs=np.ones((N,),dtype=dtype)+dC #units of e/Ec=160 [fmF]/Ec[microeV]
        

        # self.neff=self.neff_f(self.n0)
        self.kBT=kB*T #units of eV
        self.u=self.Ec/(self.kBT)
        self.Gt=Gt
        self.dtype=dtype

        self.gi=gi


        
        d1=np.concatenate((self.Cs,np.zeros((1,))))
        dm1=np.concatenate((self.Cs,np.zeros((1,))))
        d2=np.concatenate((self.second_order_C[0:-1],np.zeros((2,))))
        dm2=np.concatenate((self.second_order_C[1::],np.zeros((2,))))

        dm1=np.roll(dm1,-1)
        dm2=np.roll(dm2,-1)
        self.ntot=[sum(self.n0)]
        self.n_history=[self.n0]
        d0=np.roll(dm2,2)+np.roll(dm1,1)+np.roll(d1,-1)+np.roll(d2,-2)+np.concatenate((self.offset_C,np.zeros((2,))))
        
        data=np.array([-dm2,-dm1,d0,-d1,-d2])
        data=data[:,0:-2]
        offsets=np.array([-2,-1,0,1,2])
        self.C=sparse.dia_matrix((data,offsets),shape=(N-1,N-1),dtype=dtype)
        self.Cinv=inv(sparse.csc_matrix(self.C)).toarray()
        # potentials=self.Cinv@np.array([self.n],dtype=dtype).T

        # potentials=np.array(potentials,dtype='float64')
        # potentials=list(potentials.flatten())

        # potentials.append(self.U/(2*self.Ec))
        # potentials.append(-self.U/(2*self.Ec))
        # self.potentials=np.roll(np.array(potentials,dtype=dtype),1)
        dataM=np.array([[-1]*self.N,[1]*self.N])
        offsetsM=np.array([0,1])
        self.M=sparse.dia_matrix((dataM,offsetsM),shape=(self.N-1,self.N),dtype=dtype).toarray()
        self.MM=np.concatenate((self.M,-self.M),axis=1)
        
        self.update_number_of_concurrent(number_of_concurrent)
    def update_number_of_concurrent(self,number_of_concurrent):
        self.number_of_concurrent=number_of_concurrent
        self.MMM=np.tile(self.MM,(1,number_of_concurrent*self.number_of_Us))
        self.MMM_withoutU=np.tile(self.MM,(1,number_of_concurrent))
        # self.A=self.MMM.T@self.Cinv
        self.B=self.Cinv@self.MMM
        self.BB=self.Cinv@self.MM
        self.B_withoutU=self.Cinv@self.MMM_withoutU
        
        C=np.einsum('ij,ij->j',self.MMM,self.B)/2
        bound=np.kron(self.U/(self.Ec),(self.Cs[0]*self.B_withoutU[0,:]-self.Cs[-1]*self.B_withoutU[-1,:]+self.second_order_C[1]*self.B_withoutU[1,:]-self.second_order_C[-1]*self.B_withoutU[-2,:]))
        self.boundary_works=self.U.repeat(number_of_concurrent)/(2*self.Ec)
        self.dE0=C+bound
    

    def dE_f(self,nn):
        # v=self.A@nn 
        #np.einsum('ij,ik->kj',self.BB,nn).flatten()
        v=(nn.T@self.BB).flatten()
        # v=np.einsum('ij,ij->j',nn,self.B)
        E=v+self.dE0#+ww.flatten() #units of Ec
        # E[::2*self.N]=E[::2*self.N]+self.U/(2*self.Ec)
        # E[self.N-1::2*self.N]=E[self.N-1::2*self.N]+self.U/(2*self.Ec)
        # E[self.N::2*self.N]=E[self.N::2*self.N]-self.U/(2*self.Ec)
        # E[2*self.N-1::2*self.N]=E[2*self.N-1::2*self.N]-self.U/(2*self.Ec)
        E[::2*self.N]=E[::2*self.N]+self.boundary_works
        E[self.N-1::2*self.N]=E[self.N-1::2*self.N]+self.boundary_works
        E[self.N::2*self.N]=E[self.N::2*self.N]-self.boundary_works
        E[2*self.N-1::2*self.N]=E[2*self.N-1::2*self.N]-self.boundary_works
        return E

    def iterable(self,m):
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


    def update_transition_rate(self,n1,update_gammas=True):

        limit1=1e-12
        limit2=1e12

        dE=-self.dE_f(n1)

        if dE.ndim==1:
            c1=-dE*self.u>np.log(limit1)
            c2=-dE*self.u<np.log(limit2)
            c5=-dE*self.u<=np.log(limit1)
            c6=-dE*self.u>=np.log(limit2)
            dE1=dE[(c1) & (c2)]
            try:
                Gamma=np.array(self.gammas)#np.zeros_like(dE)
            except AttributeError:
                Gamma=np.zeros_like(dE)
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
            return Gamma.astype(self.dtype)
        
        elif self.iterable(dE)==False:
            
            if (-dE*self.u>np.log(limit1)) and (-dE*self.u<np.log(limit2)):
                Gamma=dE/(1-np.exp(-dE*self.u))
            elif (-dE*self.u<=np.log(limit1)):
                Gamma=dE
            elif (-dE*self.u>=np.log(limit2)):
                Gamma=-dE*np.exp(dE*self.u)
            # Gamma=Gamma
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
            
        if update==False:
            # try:
                if self.number_of_concurrent*self.number_of_Us==1:
                        
                    try:
                        self.p=p/Norm
                    except FloatingPointError:
                        print('probability not normalized due to floting point error')
                    return self.p
                else:
                    
                    self.p=np.einsum('ij,i->ij',self.gammas2,1/self.Gamsum)

                    return self.p
            # except Exception:
            #     if self.number_of_concurrent==1:
            #         p=self.update_transition_rate(self.Q(n),self.Q0(n))
            #         self.p=p/sum(p)
            #         return self.p
            #     else:
            #         raise Exception('Exception occured')
        else:
            if self.number_of_concurrent==1:
                # p=self.update_transition_rate(self.Q(n),self.Q0(n))
                self.p=p/Norm
                return self.p
            else:
                print('update has to be done manually when number of concurrent is greater than 1')
                self.p=np.einsum('ij,i->ij',self.gammas2,1/self.Gamsum)
                return self.p


    def pick_event(self,n,k):
        """
        

        Parameters
        ----------
        n : current charge configuration
            DESCRIPTION.
        k : number of events
            DESCRIPTION.

        Returns
        -------
        index : (k,)-array of indices of events chosen
            DESCRIPTION.

        """
        if self.number_of_concurrent==1:
            
            index=pick_event2(np.array([self.p]))
            
            # index=random.choices(np.arange(2*self.N),weights=self.p,k=k)
            
            return index[0]
        else:
            indices=[]

            for s in np.arange(self.number_of_concurrent):
                    
                # indices.append(random.choices(np.arange(2*self.N)+2*self.N*s,weights=self.p[s],k=k))
                indices.append(random.choices(np.arange(2*self.N),weights=self.p[s],k=k))
            return indices
    
    
        
    def dt_f(self,n):
        factor_SI=e_SI/(self.N*self.Ec*self.Gt)
        if self.number_of_concurrent==1:
            try:
                # self.dts=factor_SI/(self.gammas[0:self.N]+self.gammas[self.N::])
                self.dts=factor_SI/self.sumgammas
                return self.dts
                # return sum(self.dts)
            except Exception:
                self.update_transition_rate(n)
                # self.dts=factor_SI/(self.gammas[0:self.N]+self.gammas[self.N::])
                return self.dts
                self.dts=factor_SI/self.sumgammas
                # return sum(self.dts)
        else:
                self.dts=factor_SI/self.Gamsum
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

    def ntot_f(self):
        self.ntot.append(np.sum(self.n,axis=0))
    def store_n(self):
        self.n_history.append(self.n)
    def plot_event_histograms(self,n,samples=None):
        if samples is None:
            samples=100*self.N
        indices=np.array(self.pick_event(n,samples))
        indices_rl=indices[indices<self.N]
        indices_lr=indices[indices>=self.N]-self.N
        plt.figure()
        plt.hist(indices_rl,density=True,bins=self.N,label='total # of events={}'.format(len(indices_rl)))
        pr=sum(self.P(n)[0:self.N])/sum(self.P(n))
        plt.plot(self.P(n)[0:self.N]/pr,label='renormalized P of right moving. (P(right)={:.3f} pct.)'.format(pr*100))
        plt.legend()
        plt.xlabel('site number')
        plt.figure()
        plt.hist(indices_lr,density=True,bins=self.N,label='total # of events={}'.format(len(indices_lr)))
        pl=sum(self.P(n)[self.N::])/sum(self.P(n))
        plt.plot(self.P(n)[self.N::]/pl,label='renormalized P of left moving. (P(left)={:.3f} pct.)'.format(pl*100))
        plt.legend()
        plt.xlabel('site number')
    def update_potentials(self):
        v=self.Cinv@np.array([self.n]).T
        self.potentials[1:-1]=v.flatten()
        
        
    def step(self,store_data=False):
        neff=self.n#self.neff_f(self.n)
        # Qneff=self.Q(neff)

        self.update_transition_rate(neff)
        self.P(neff)
        self.index=self.pick_event(neff,1)
        # Q_new=Q0neff+self.MM#self.Q(self.n)
        n_new=neff+self.MM[:,self.index[0]]#Q_new[:,self.index[0]]
        if store_data:
            self.dtp.append(self.dt_f(neff))
            self.dQp.append(self.dQ_f(neff))
            # dt,dQ=self.dt_and_dQ_2_f(Qneff)
            # self.dtp2.append(dt)
            # self.dQp2.append(dQ)
        self.n=n_new
        if store_data:
            self.store_n()
            self.ntot_f()
    def initiate_multistep(self,randomize_initial=False):
        
        self.ns=np.array([self.n0]*self.number_of_concurrent)
        if randomize_initial==True:
            for j in np.arange(self.number_of_concurrent-1):
                
                self.ns[j+1]=self.n0+np.array(random.choices(np.arange(11)-5,k=N-1,weights=np.exp(-0.2*(np.arange(11)-5)**2)))
        self.ns=self.ns.T
        return self.ns

    def multistep(self,store_data=False):
        # with np.errstate(divide='raise'):
            try:
                neff=self.ns#self.neff_fnD(self.ns)
                # self.QQ_nD(neff)
                # self.Q0Q_nD(neff)
                self.gammas2=self.update_transition_rate(neff).reshape(self.number_of_concurrent*self.number_of_Us,2*self.N)
                self.Gamsum=np.sum(self.gammas2,axis=1)
                self.Gamdif=np.sum(self.gammas2[:,0:self.N]-self.gammas2[:,self.N::],axis=1)
                self.P(neff)
                if store_data:
                    
                    self.dtp.append(self.dt_f(neff))
                    self.dQp.append(self.dQ_f(neff))
        
                    # self.Ip.append(np.sum(np.array(self.dQp))/np.sum(np.array(self.dtp)))
                self.indices=pick_event2(self.p)#self.pick_event(neff,1)
                self.indices=list(self.indices)
                # print(self.indices)
                # Q_new=self.Q0Qn+self.MMM#self.QQ_nD(self.ns,onlyQ=True)
                n_new=neff+self.MM[:,self.indices]#[:,:,0]
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
        self.AutocorI=[]
        if number_of_concurrent==1:
            self.number_of_concurrent=number_of_concurrent
            if print_every is None:
                print_every=int(number_of_steps/100+1)
            
            for i in np.arange(number_of_steps):
                if print_every != 0:
                    if i%print_every==print_every-1:
                        print('{:.1f}'.format(i*100/number_of_steps)+' pct.')
    
                        self.step(store_data=True)
                    else:
                        self.step(store_data=False)
                else:
                    self.step(store_data=False)
            self.final_current=np.sum(np.array(self.dQp)[transient::])/np.sum(np.array(self.dtp)[transient::])
            return self.final_current
        else:

            if skip_transient:
                print('n0 is being moved forward through the transient window')

                self.update_number_of_concurrent(1)
                
                self.ns=np.array([self.n0]*self.number_of_Us).T
                for j in np.arange(transient*print_every):
                    self.multistep(store_data=False)
                    
                # self.n0=np.array(self.n)
                
                del self.gammas
                del self.p

            self.update_number_of_concurrent(number_of_concurrent)

            print('initiating multistep')
            
            # self.initiate_multistep(randomize_initial=False)
            print(self.ns)
            self.ns=self.ns.repeat(self.number_of_concurrent,axis=1)
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
            if skip_transient:
                self.final_currents=np.sum(np.array(self.dQp),axis=0)/np.sum(np.array(self.dtp),axis=0)
            else:
                self.final_currents=np.sum(np.array(self.dQp)[transient::,:],axis=0)/np.sum(np.array(self.dtp)[transient::,:],axis=0)
                
            return self.final_currents
        

class conductance:
    
    def __init__(self,N,T,Ec,Gt,offset_C=None,second_order_C=None,n0=None,gi=None,U=0,dtype='float64',number_of_concurrent=1,dC=None,q0=0,skip_transient=True):
        if dC is None:
            dC=0
            self.dC=dC
        else:
            self.dC=dC
        if gi is None:
            gi=1
            self.gi=gi
        else:
            self.gi=gi
        if n0 is None:
            n0=np.array([0]*(N-1))
        elif n0=='random':
            n0=np.array(random.choices(np.arange(11)-5,k=N-1,weights=np.exp(-0.2*(np.arange(11)-5)**2)))

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
        self.n0=n0.astype(dtype)+q0 #units of number of electrons
        self.n=n0.astype(dtype)
        self.Cs=np.ones((N,),dtype=dtype)+dC #units of e/Ec=160 [fmF]/Ec[microeV]
        self.q0=q0
        self.U=U #units of eV
        
        # self.neff=self.neff_f(self.n0)
        self.T=T
        self.kBT=kB*T #units of eV
        self.u=self.Ec/(self.kBT)
        self.Gt=Gt
        self.dtype=dtype
        self.gi=gi
        self.number_of_concurrent=number_of_concurrent
        self.skip_transient=skip_transient
        # self.CBT=CBTmontecarlo(N,T,Ec,Gt,offset_C,second_order_C,n0,gi,U,dtype,number_of_concurrent,dC)
            
    def f(self,U):
        N=self.N
        T=self.T
        Ec=self.Ec
        Gt=self.Gt
        offset_C=self.offset_C
        second_order_C=self.second_order_C
        n0=self.n0
        gi=self.gi
        dtype=self.dtype
        number_of_concurrent=self.number_of_concurrent
        dC=self.dC
        
        
        self.CBT=CBTmontecarlo(N,T,Ec,Gt,offset_C,second_order_C,n0,gi,U,dtype,number_of_concurrent,dC)
        

        current=self.CBT(self.number_of_steps,self.transient,print_every=self.store_interval,number_of_concurrent=number_of_concurrent,skip_transient=self.skip_transient)
        if number_of_concurrent>1:
            dQ=np.array(self.CBT.dQp)[transient::,:]
            dt=np.array(self.CBT.dtp)[transient::,:]
        else:
            dQ=np.array(self.CBT.dQp)[transient::]
            dt=np.array(self.CBT.dtp)[transient::]
        #sigI=np.sqrt(np.var(dQ,axis=0)/np.sum(dt,axis=0)**2+(np.sum(dQ,axis=0)/np.sum(dt,axis=0))**2*np.var(dt,axis=0)/np.sum(dt,axis=0)**2)*np.sqrt(number_of_steps/print_every-transient)

        return current,dQ,dt
    def f2(self,U):
        N=self.N
        T=self.T
        Ec=self.Ec
        Gt=self.Gt
        offset_C=self.offset_C
        second_order_C=self.second_order_C
        n0=self.n0
        gi=self.gi
        dtype=self.dtype
        number_of_concurrent=self.number_of_concurrent
        dC=self.dC
        
        
        self.CBT=CBTmontecarlo(N,T,Ec,Gt,offset_C,second_order_C,n0,gi,U,dtype,number_of_concurrent,dC)
        

        current=self.CBT(self.number_of_steps,self.transient,print_every=self.store_interval,number_of_concurrent=number_of_concurrent,skip_transient=self.skip_transient)
        if number_of_concurrent>1:
            dQ=np.array(self.CBT.dQp)[transient::,:]
            dt=np.array(self.CBT.dtp)[transient::,:]
        else:
            dQ=np.array(self.CBT.dQp)[transient::]
            dt=np.array(self.CBT.dtp)[transient::]
        #sigI=np.sqrt(np.var(dQ,axis=0)/np.sum(dt,axis=0)**2+(np.sum(dQ,axis=0)/np.sum(dt,axis=0))**2*np.var(dt,axis=0)/np.sum(dt,axis=0)**2)*np.sqrt(number_of_steps/print_every-transient)

        return current,dQ,dt   
    def run(self,Us,number_of_steps,store_interval,transient,T=None,number_of_concurrent=None,n_jobs=4,parallelize=True):
        if number_of_concurrent is None:
            number_of_concurrent=self.number_of_concurrent
        else:
            self.number_of_concurrent=number_of_concurrent
        if T is None:
            T=self.T
        else:
            self.T=T 
        self.number_of_steps=number_of_steps
        self.store_interval=store_interval
        self.transient=transient
        self.Us=Us
        self.now=str(datetime.now()).replace(':','.')
        a=time()
        if parallelize:
            Is=Parallel(n_jobs=n_jobs,verbose=50)(delayed(self.f)(U) for U in Us)   
        else:
            Is=[]
            for U in Us:
                Is.append(self.f(U))
        b=time()
        self.simulation_time=b-a
        self.Is=Is
        self.currents=np.array([I[0] for I in Is])
        self.dQ=np.array([I[1] for I in Is])
        self.dt=np.array([I[2] for I in Is])
        if number_of_concurrent>1:
            self.currentsm=np.mean(self.currents,axis=1)
        else:
            self.currentsm=np.mean(self.currents)
        self.points=int(len(self.currentsm)/2)
        points=self.points
        self.Gs=(self.currents[points:2*points,:]-self.currents[0:points,:])/np.array([(Us[points:2*points]-Us[0:points])]).repeat(len(self.currents[0,:]),axis=0).T
        self.Gsm=np.sum(self.Gs,axis=1)/len(self.Gs[0,:])
        self.Gstd=np.std(self.Gs,axis=1)
        return self.Gsm,self.Gstd,self.Gs,Is
    def run2(self,Us,number_of_steps,store_interval,transient,T=None,number_of_concurrent=None,parallelize=True):
        if number_of_concurrent is None:
            number_of_concurrent=self.number_of_concurrent
        else:
            self.number_of_concurrent=number_of_concurrent
        if T is None:
            T=self.T
        else:
            self.T=T 
        self.number_of_steps=number_of_steps
        self.store_interval=store_interval
        self.transient=transient
        self.Us=Us
        self.now=str(datetime.now()).replace(':','.')
        a=time()

        current,dQ,dt=self.f(Us)

        b=time()
        self.simulation_time=b-a
        # self.Is=Is
        self.currents=np.array([current[self.number_of_concurrent*s:self.number_of_concurrent*(s+1)] for s in np.arange(len(Us))])
        self.dQ=np.array([dQ[:,self.number_of_concurrent*s:self.number_of_concurrent*(s+1)] for s in np.arange(len(Us))])
        self.dt=np.array([dt[:,self.number_of_concurrent*s:self.number_of_concurrent*(s+1)] for s in np.arange(len(Us))])
        if number_of_concurrent>1:
            self.currentsm=np.mean(self.currents,axis=1)
        else:
            self.currentsm=np.mean(self.currents)
        self.points=int(len(self.currentsm)/2)
        points=self.points
        self.Gs=(self.currents[points:2*points,:]-self.currents[0:points,:])/np.array([(Us[points:2*points]-Us[0:points])]).repeat(len(self.currents[0,:]),axis=0).T
        self.Gsm=np.sum(self.Gs,axis=1)/len(self.Gs[0,:])
        self.Gstd=np.std(self.Gs,axis=1)
        Is=None
        self.Is=Is
        return self.Gsm,self.Gstd,self.Gs,Is
    def auto(self,x,step):
        xm=np.mean(x)
        A=(np.roll(x,step)-xm)*(x-xm)
        A=A[step::]
        num=np.mean(A)/np.var(x)
        return num
    def CBT_model_g(self,x):
        return (x*np.sinh(x)-4*np.sinh(x/2)**2)/(8*np.sinh(x/2)**4)

    def CBT_model_G(self,V):
        return self.Gt*(1-self.Ec*self.CBT_model_g(V/(self.N*kB*self.T))/(kB*self.T))
    def plotG(self,save=False):

        points=self.points
        fig,ax=plt.subplots(figsize=(9,6))
        plt.title('MC for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={:.1e}e, steps between samples={}, \n steps/(run'.format(self.N,self.T*1e3,self.Ec*1e6,self.Gt*1e6,self.q0,
                                                                                                                                                            self.store_interval)+r'$\times$'+'datapoint)={}, runs/datapoint={}, transient interval={}'.format(self.number_of_steps,self.number_of_concurrent,self.transient*self.store_interval))
        
        Us=self.Us
        Vs=(Us[points:2*points]+Us[0:points])/2
        for i in np.arange(len(self.Gs[0,:])):
            plt.plot(Vs,self.Gs[:,i],'.',color=[(i+1)/len(self.Gs[0,:]),0,0])
        ax.plot(np.linspace(min(Us),max(Us),1000),self.CBT_model_G(np.linspace(min(Us),max(Us),1000)),label='First order analytic result')
        ax.set_xlabel('Bias voltage [V]')
        ax.set_ylabel(r'Differential Conductance [Si], ($\Delta$ I/$\Delta$ V, $\Delta$ V={:.1e} eV)'.format(2*self.dV))
        ax.legend()
        plt.grid()
        plt.tight_layout()
        fig2,ax2=plt.subplots(figsize=(9,6))

        plt.title('Results for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={:.1e}e'.format(self.N,self.T*1e3,self.Ec*1e6,self.Gt*1e6,self.q0))
        ax2.errorbar(Vs,self.Gsm,yerr=self.Gstd/np.sqrt(len(self.Gs[0,:])),fmt='.',label='Monte Carlo simulation results (mean)')
        ax2.plot(np.linspace(min(Us),max(Us),1000),self.CBT_model_G(np.linspace(min(Us),max(Us),1000)),label='first order analytic result')
        ax2.set_xlabel('bias voltage [V]')
        ax2.set_ylabel(r'Differential Conductance [Si], ($\Delta$ I/$\Delta$ V, $\Delta$ V={:.1e} eV)'.format(2*self.dV))
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
        
    def __call__(self,V,number_of_steps,store_interval,transient,T=None,number_of_concurrent=None,n_jobs=4,dV=None,split=True,plot=True,save_data=True,parallelize=True,internal_parallelization=True):
        self.Vhalf=5.439*kB*self.T*self.N
        try:
            if split:
                if dV is None:
                    self.dV=self.Vhalf/60
                else:
                    self.dV=dV
    
                Us=V-self.dV
                Us=np.concatenate((Us,V+self.dV))
            else:
                Us=V
            if internal_parallelization:
                self.run2(Us,number_of_steps,store_interval,transient,T,number_of_concurrent,parallelize)
            else:
                self.run(Us,number_of_steps,store_interval,transient,T,number_of_concurrent,n_jobs,parallelize)
            if plot:
                self.plotG(save=True)
            return self.Gsm,self.Gstd,self.Gs,self.Is
        except ValueError:
            if number_of_concurrent is None:
                number_of_concurrent=self.number_of_concurrent
            else:
                self.number_of_concurrent=number_of_concurrent
            if T is None:
                T=self.T
            else:
                self.T=T 
            self.number_of_steps=number_of_steps
            self.store_interval=store_interval
            self.transient=transient
            self.Us=V
            self.Is=self.f(V)
            return self.Is
            
        
        
#%%
if __name__=='__main__':
#-np.ones((N-1,))*10
    # array([ 513.,    0.,   -2.,    2.,    1.,    1.,    0.,   -2.,    3.,
    #          -1.,    1.,   -1.,   -1.,   -3.,    2.,    4.,   -4.,    5.,
    #          -5.,   -1.,    3.,    0.,   -3.,    6.,   -8.,    7.,   -2.,
    #          -5.,    7.,   -5.,    3.,   -1.,    2.,    1.,   -4.,    1.,
    #           0.,   -2.,    7.,   -2.,   -7.,    0.,    1.,   -1.,    3.,
    #           1.,   -9.,    8.,    4.,   -9.,    8.,   -3.,    1.,   -1.,
    #           1.,    0.,   -3.,    0.,    7.,   -3.,    1.,   -3.,   -3.,
    #           4.,   -4.,    1.,    1.,    1.,    2.,   -2.,    5.,   -1.,
    #          -4.,    2.,    2.,    2.,   -4.,    3.,    1.,   -2.,   -7.,
    #           1.,    1.,    3.,    1.,   -3.,   -2.,    0.,    6.,   -1.,
    #          -3.,    2.,    2.,   -4.,   -2.,    4.,    6.,   -4., -514.])
    # n0[40:60]=n0[40:60]+100
    # Cs=np.ones((N,))
    # offset_C=-0*np.ones((N-1,))*1e-4
    # offset_q=0*n0/N
    # second_order_C=0*Cs*1e-1
    N=100
    n0=0*np.array(random.choices(np.arange(11)-5,k=N-1,weights=np.exp(-0.2*(np.arange(11)-5)**2)))
    Ec=4e-6
    Gt=2e-5
    gi=np.ones((2*N,))
    T=0.01
    FWHM=5.439*kB*T*N
    q0=0
    # points=21
    points=5
    lim=3*FWHM
    dV=FWHM/50
    Vs=np.linspace(-lim,lim,points)
    # Vs=np.concatenate((Vs,np.linspace(-8*lim/(points),8*lim/(points),8)))

    number_of_steps=600
    transient=4000
    print_every=10
    number_of_concurrent=100
    
    # sim=CBTmontecarlo(N,T,Ec,Gt,U=Vs)
    # sim(100,10,2,number_of_concurrent=3)
    gg=conductance(N,T,Ec,Gt,n0=n0,skip_transient=True)

    Gsm,Gstd,Gs,Is=gg(Vs,number_of_steps=number_of_steps,
                      transient=transient,
                      store_interval=print_every,
                      number_of_concurrent=number_of_concurrent,
                      n_jobs=2)
    
    # def plotG():
    #     plt.figure()
    #     plt.title('Results for T={:.1e} mK, Ec={:.1e} $\mu$eV, Gt={:.1e} $\mu$Si, q0={:.1e}e'.format(T*1e3,Ec*1e6,Gt*1e6,q0))
        
    #     Vs=(Us[points:2*points]+Us[0:points])/2
    #     for i in np.arange(len(Gs[0,:])):
    #         plt.plot(Vs,Gs[:,i],'.',label='Monte Carlo simulation results',color=[(i+1)/len(Gs[0,:]),0,0])
    #     plt.plot(np.linspace(Us[0],Us[-1],1000),CBT_model_G(np.linspace(Us[0],Us[-1],1000)),label='first order analytic result')
    #     plt.xlabel('bias voltage [V]')
    #     plt.ylabel('Differential Conductance [Si]')
    #     plt.legend()
    #     plt.grid()

    #     plt.figure()

    #     plt.title('Results for T={:.1e} mK, Ec={:.1e} $\mu$eV, Gt={:.1e} $\mu$Si, q0={:.1e}e'.format(T*1e3,Ec*1e6,Gt*1e6,q0))
    #     plt.errorbar(Vs,Gsm,yerr=Gstd,fmt='.',label='Monte Carlo simulation results')
    #     plt.plot(np.linspace(Us[0],Us[-1],1000),CBT_model_G(np.linspace(Us[0],Us[-1],1000)),label='first order analytic result')
    #     plt.xlabel('bias voltage [V]')
    #     plt.ylabel('Differential Conductance [Si]')
    #     plt.legend()
    #     plt.grid()
    # CBT=CBTmontecarlo(N,offset_q,n0,U,T,Cs,offset_C,second_order_C,Ec,Gt,gi,dtype='float64')
    # number_of_steps=10000
    # transient=2
    # current=CBT(number_of_steps,transient,print_every=500,number_of_concurrent=50)
    
    # b=time()
    # print(b-a)
    
    # def timing(n):
    #     print(n)
    #     a=time()

    #     number_of_steps=2000
    #     transient=1
    #     print_every=100
    #     number_of_concurrent=n
    #     gg=conductance(N,T,Ec,Gt,n0=n0,skip_transient=True)
    #     gg(Vs[0],number_of_steps=number_of_steps,
    #                       transient=transient,
    #                       store_interval=print_every,
    #                       number_of_concurrent=number_of_concurrent,
    #                       n_jobs=4)
    #     b=time()
    #     return b-a
    # times=[]
    # ns=50*np.arange(4)+1
    # for n in ns:
    #     times.append(timing(n))
    # plt.figure()
    # plt.plot(ns,times)
    # from scipy.optimize import curve_fit
    # def lin(x,a,b):
    #     return a*x+b
    # par,cov=curve_fit(lin,ns,times)
    # plt.plot(ns,lin(ns, *par),label='{:.1f}s/n'.format(par[0]))
    
    # a=time()
    # def f(U):
        
    #     CBT=CBTmontecarlo(N,offset_q,n0,U,T,Cs,offset_C,second_order_C,Ec,Gt,gi,dtype='float64')
    #     number_of_steps=50000
    #     transient=2
    #     print_every=100
    #     current=CBT(number_of_steps,transient,print_every=print_every)
    #     dQ=np.array(CBT.dQp)[transient::]
    #     dt=np.array(CBT.dtp)[transient::]
    #     sigI=np.sqrt(np.var(dQ)/np.sum(np.array(dt))**2+(np.sum(dQ)/np.sum(dt))**2*np.var(dt)/np.sum(dt)**2)*np.sqrt(number_of_steps/print_every-transient)
        
    #     current2=np.sum(np.array(CBT.dQp2)[transient::])/np.sum(np.array(CBT.dtp2)[transient::])
    #     dQ=np.array(CBT.dQp2)[transient::]
    #     dt=np.array(CBT.dtp2)[transient::]
    #     sigI2=np.sqrt(np.var(dQ)/np.sum(np.array(dt))**2+(np.sum(dQ)/np.sum(dt))**2*np.var(dt)/np.sum(dt)**2)*np.sqrt(number_of_steps/print_every-transient)
    #     return current,current2,sigI,sigI2
    
    # number_of_steps=4000
    # transient=5
    # print_every=200
    # def f(U):
        
    #     CBT=CBTmontecarlo(N,n0,U,T,Cs,offset_C,second_order_C,Ec,Gt,gi,dtype='float64',number_of_concurrent=100)

    #     current=CBT(number_of_steps,transient,print_every=print_every,number_of_concurrent=100)
    #     dQ=np.array(CBT.dQp)[transient::,:]
    #     dt=np.array(CBT.dtp)[transient::,:]
    #     #sigI=np.sqrt(np.var(dQ,axis=0)/np.sum(dt,axis=0)**2+(np.sum(dQ,axis=0)/np.sum(dt,axis=0))**2*np.var(dt,axis=0)/np.sum(dt,axis=0)**2)*np.sqrt(number_of_steps/print_every-transient)
        

    #     return current,dQ,dt
    
    # Is=Parallel(n_jobs=4,verbose=50)(delayed(f)(U) for U in Us)

    # currents=np.array([I[0] for I in Is])
    # dcurrents=np.array([I[1] for I in Is])
    # dQ=np.array([I[2] for I in Is])
    # dt=np.array([I[3] for I in Is])
    # dcurrents=np.array([I[1] for I in Is])
    # plt.figure()
    # for j in np.arange(len(currents[0,:])):
    #     plt.errorbar(Us,currents[:,j],yerr=dcurrents[:,j]/np.sqrt(len(dcurrents[0,:])),fmt='.',color=[0,0,j/len(dcurrents[0,:])])
    # currentsm=np.mean(currents,axis=1)
    
    # gm1=(currentsm[points:2*points]-currentsm[0:points])/(Us[points:2*points]-Us[0:points])
    
    # b=time()
    # print(b-a)
    
    # def CBT_model_g(x):
    #     return (x*np.sinh(x)-4*np.sinh(x/2)**2)/(8*np.sinh(x/2)**4)



    # def CBT_model_G(V):
        
    #     return Gt*(1-Ec*CBT_model_g(V/(N*kB*T))/(kB*T))

    # plt.figure()
    # plt.errorbar(Us[points:2*points],gm1,fmt='.')
    # plt.plot(np.linspace(Us[0],Us[-1],1000),CBT_model_G(np.linspace(Us[0],Us[-1],1000)))
    

    # for U in 
    # CBT.plot_event_histograms(CBT.n0)
    # plt.figure()
    # for i in np.arange(number_of_steps):
    #     if i%100==0:
    #         print(i)
    #         plt.plot(CBT.n,color=[0,0,i/number_of_steps])
    #     CBT.step()
    
    # CBT.plot_event_histograms(CBT.n_history[0])
    # plt.figure()
    # plt.plot(CBT.ntot)
    # plt.ylabel('charge')
    # plt.figure()
    # plt.ylabel('probability')
    # for s in np.arange(len(CBT.n_history)):
    #     plt.plot(CBT.P(CBT.n_history[s],update=True),color=[0,0,s/len(CBT.n_history)])
    # plt.figure()
    # plt.ylabel('charge')
    # s=1
    # plt.plot(CBT.n_history[s],color=[0,0,s/len(CBT.n_history)])
    # plt.figure()
    # plt.plot(np.array(CBT.dtp))
    # plt.figure()
    # plt.plot(np.array(CBT.dQp))
    # plt.figure()
    # plt.plot(np.array(CBT.dQp)[100::]/np.array(CBT.dtp)[100::])
    # # plt.figure()
    # # plt.hist(CBT.pick_event(CBT.n0,10000),density=True,bins=2*CBT.N)
    # # plt.plot(CBT.P(CBT.n0))