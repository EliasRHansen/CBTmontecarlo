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
#########Input
class CBTmontecarlo:
    
    def __init__(self,N,offset_q,n0,U,kBT,Cs,offset_C,second_order_C):
        self.N=N
        self.n0=n0
        self.Cs=Cs
        self.second_order_C=second_order_C
        self.U=U
        self.offset_q=offset_q
        self.offset_C=offset_C
        self.n0eff=self.neff(self.n0)
        # self.n0eff=copy(self.n0)
        # self.n0eff[0]=self.n0eff[0]-self.Cs[0]*self.U/2
        # self.n0eff[1]=self.n0eff[1]-self.second_order_C[1]*self.U/2
        # self.n0eff[-1]=self.n0eff[-1]+self.Cs[-1]*self.U/2
        # self.n0eff[-2]=self.n0eff[-2]+self.second_order_C[-1]*self.U/2
        self.kBT=kBT
        d1=np.concatenate((self.Cs,np.zeros((1,))))
        dm1=np.concatenate((self.Cs,np.zeros((1,))))
        d2=np.concatenate((self.second_order_C[0:-1],np.zeros((2,))))
        dm2=np.concatenate((self.second_order_C[1::],np.zeros((2,))))
        
        dm1=np.roll(dm1,-1)
        dm2=np.roll(dm2,-1)
        
        d0=np.roll(dm2,2)+np.roll(dm1,1)+np.roll(d1,-1)+np.roll(d2,-2)+np.concatenate((self.offset_C,np.zeros((2,))))
        
        data=np.array([dm2,dm1,d0,d1,d2])
        data=data[:,0:-2]
        offsets=np.array([-2,-1,0,1,2])
        self.C=sparse.dia_matrix((data,offsets),shape=(N-1,N-1),dtype='float64')
        self.Cinv=inv(sparse.csc_matrix(self.C))
        
        dataM=np.array([[-1]*self.N,[1]*self.N])
        offsetsM=np.array([0,1])
        self.M=sparse.dia_matrix((dataM,offsetsM),shape=(self.N-1,self.N),dtype='float64').toarray()
    
    def neff(self,n):
        self.neff=copy(n)
        self.neff[0]=self.neff[0]-self.Cs[0]*self.U/2
        self.neff[1]=self.neff[1]-self.second_order_C[1]*self.U/2
        self.neff[-1]=self.neff[-1]+self.Cs[-1]*self.U/2
        self.neff[-2]=self.neff[-2]+self.second_order_C[-1]*self.U/2
        return self.neff
    
    def energy(self,n):
        if n.shape==(self.N-1,):
            v=self.Cinv@np.array([n]).T
            boundaries=(self.Cs[0]*v[0]+self.second_order_C[1]*v[1]-self.Cs[-1]*v[-1]-self.second_order_C[-1]*v[-2])*self.U/2
            return np.sum(n*(v.flatten()/2+self.offset_q))+boundaries[0]
        elif n.shape==(self.N-1,2*self.N):
            v=self.Cinv@n
            w=np.einsum('ij,ij->j',n,v)     
            ww=n.T@np.array([self.offset_q]).T
            boundaries=(self.Cs[0]*v[0,:]+self.second_order_C[1]*v[1,:]-self.Cs[-1]*v[-1,:]-self.second_order_C[-1]*v[-2,:])*self.U/2
            return w.flatten()/2+ww.flatten()+boundaries
        else:
            raise Exception('energy could not be calculated due to incorrect shape of charge array')

    def iterable(self,m):
        try:
            iter(m)
            return True
        except TypeError:
            return False

    def Q(self,n):
        Qr=np.array([n]).T.repeat(self.N,axis=1)+self.M
        Ql=np.array([n]).T.repeat(self.N,axis=1)-self.M
        return np.concatenate((Qr,Ql),axis=1)
    def Q0(self,n):
        Qr=np.array([n]).T.repeat(self.N,axis=1)
        return np.concatenate((Qr,Qr),axis=1)

    def transition_rate(self,n2,n1):
        
        dE=self.energy(n2)-self.energy(n1)
        limit1=1e-15
        limit2=1e15
        if dE.shape==(2*self.N,):
            Gamma=np.zeros_like(dE)
            Gamma[(-dE/self.kBT>np.log(limit1)) & (-dE/self.kBT<np.log(limit2))]=dE[(-dE/self.kBT>np.log(limit1)) & (-dE/self.kBT<np.log(limit2))]/(1-np.exp(-dE[(-dE/self.kBT>np.log(limit1)) & (-dE/self.kBT<np.log(limit2))]/self.kBT))
            Gamma[-dE/self.kBT<=np.log(limit1)]=dE[-dE/self.kBT<=np.log(limit1)]
            Gamma[-dE/self.kBT>=np.log(limit2)]=dE[-dE/self.kBT>=np.log(limit2)]*np.exp(dE[-dE/self.kBT>=np.log(limit2)]/self.kBT)
            return Gamma
        elif self.iterable(dE):
            
            if (-dE/self.kBT>np.log(limit1)) and (-dE/self.kBT<np.log(limit2)):
                Gamma=dE/(1-np.exp(-dE/self.kBT))
            elif (-dE/self.kBT<=np.log(limit1)):
                Gamma=dE
            elif (-dE/self.kBT>=np.log(limit2)):
                Gamma=dE*np.exp(dE/self.kBT)
            return Gamma

    def P(self,n):
        
        p=self.transition_rate(self.Q(n),self.Q0(n))
        
        return p/sum(p)

    def pick_event(self,n,k):
        
        index=random.choices(np.arange(2*self.N),weights=self.P(n),k=k)
        
        return index


if __name__=='__main__':
    
    N=100
    test=np.linspace(1,N,N)

    n0=-np.ones((N-1,))
    Cs=np.ones((N,))
    offset_C=-np.ones((N-1,))/100
    offset_q=offset_C/10
    second_order_C=0*Cs/1000
    U=10
    kBT=1e-0
    
    CBT=CBTmontecarlo(N,offset_q,n0,U,kBT,Cs,offset_C,second_order_C)
    plt.figure()
    plt.hist(CBT.pick_event(CBT.n0,10000),density=True,bins=2*CBT.N)
    plt.plot(CBT.P(CBT.n0))
