# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 22:34:41 2021

@author: Elias Roos Hansen
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import inv
from copy import copy
import random as random
N=100
test=np.linspace(1,N,N)

n0=-np.ones((N-1,))
Cs=np.ones((N,))
offset_C=-np.ones((N-1,))/10000
offset_phi=np.ones((N-1,))/10
offset_q=offset_phi*offset_C
second_order_C=0*Cs/1000


U=10
n0eff=copy(n0)
n0eff[0]=n0eff[0]-Cs[0]*U/2
n0eff[1]=n0eff[1]-second_order_C[1]*U/2
n0eff[-1]=n0eff[-1]+Cs[-1]*U/2
n0eff[-2]=n0eff[-2]+second_order_C[-1]*U/2
# d0=Cs[0:-1]+Cs[1::]+second_order_C[0:-1]+np.roll(second_order_C[1::],-1)+offset_C
# d0[0]=d0[0]-second_order_C[0]
# d0[-1]=d0[-1]-second_order_C[-1]
d1=np.concatenate((Cs,np.zeros((1,))))
dm1=np.concatenate((Cs,np.zeros((1,))))
d2=np.concatenate((second_order_C[0:-1],np.zeros((2,))))
dm2=np.concatenate((second_order_C[1::],np.zeros((2,))))

dm1=np.roll(dm1,-1)
dm2=np.roll(dm2,-1)

d0=np.roll(dm2,2)+np.roll(dm1,1)+np.roll(d1,-1)+np.roll(d2,-2)+np.concatenate((offset_C,np.zeros((2,))))

data=np.array([dm2,dm1,d0,d1,d2])
data=data[:,0:-2]
offsets=np.array([-2,-1,0,1,2])
C=sparse.dia_matrix((data,offsets),shape=(N-1,N-1),dtype='float64')
aa=C.todense()
Cinv=inv(sparse.csc_matrix(C))
bb=Cinv.todense()

def energy(n):
    if n.shape==(N-1,):
        v=Cinv@np.array([n]).T
        return np.sum(n*(v.flatten()/2+offset_q))+(Cs[0]*v[0]+second_order_C[1]*v[1]-Cs[-1]*v[-1]-second_order_C[-1]*v[-2])*U/2
    elif n.shape==(N-1,2*N):
        v=Cinv@n
        w=np.einsum('ij,ij->j',n,v)     
        ww=n.T@np.array([offset_q]).T
        return w.flatten()/2+ww.flatten()+(Cs[0]*v[0,:]+second_order_C[1]*v[1,:]-Cs[-1]*v[-1,:]-second_order_C[-1]*v[-2,:])*U/2
    else:
        raise Exception('energy could not be calculated due to incorrect shape of charge array')
# def transfer_right(n,i):
#     nn=n
#     if (i<len(n)):
#         nn[i]=n[i]+1
#     if i>0:
#         nn[i-1]=n[i-1]-1
#     return copy(nn)
# def transfer_left(n,i):
#     nn=n
#     if (i<len(n)):
#         nn[i]=n[i]-1
#     if i>0:
#         nn[i-1]=n[i-1]+1
#     return copy(nn)

data=np.array([[-1]*N,[1]*N])
offsets=np.array([0,1])
M=sparse.dia_matrix((data,offsets),shape=(N-1,N),dtype='float64').toarray()
def Q(n):
    Qr=np.array([n]).T.repeat(N,axis=-1)+M
    Ql=np.array([n]).T.repeat(N,axis=-1)-M
    return np.concatenate((Qr,Ql),axis=1)
def Q0(n):
    Qr=np.array([n]).T.repeat(N,axis=-1)
    return np.concatenate((Qr,Qr),axis=1)

def transition_rate(n2,n1,kBT):
    
    dE=energy(n2)-energy(n1)
    
    Gamma=dE/(1-np.exp(-dE/kBT))
    return Gamma

def P(n,kBT):
    
    p=transition_rate(Q(n0),Q0(n0),kBT)
    
    return p/sum(p)

def pick_event(Ps,k):
    
    index=random.choices(np.arange(2*N),weights=Ps,k=k)
    
    return index

plt.figure()
plt.hist(pick_event(P(n0eff,1e-1),10000),density=True,bins=2*N)
plt.plot(P(n0eff,1e-1))
