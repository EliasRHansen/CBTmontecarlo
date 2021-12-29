# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 18:00:32 2021

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

#########Input

N=10
test=np.linspace(1,N,N)

n0=np.ones((N+1,))
Cs=test*np.ones((N,))
offset_C=0*np.ones((N-1,))/10
offset_phi=np.ones((N-1,))/10
offset_q=offset_phi*offset_C
second_order_C=Cs/100
U=1
n0[0]=-U/2
n0[N]=U/2


d1=Cs[1::]
d1=np.roll(np.concatenate((np.zeros((3,)),d1)),-1)
dm1=np.concatenate((Cs[0:-1],np.zeros((3,))))
d2=np.concatenate((np.zeros((3,)),second_order_C[0:-1]))
d2=np.roll(d2,-1)
dm2=np.concatenate((second_order_C[0:-1],np.zeros((3,))))
d0=np.roll(dm2,1)+dm1+np.roll(d1,-2)+np.roll(d2,-3)
d0=np.roll(d0,1)

data=np.array([dm2,dm1,d0,d1,d2])
offsets=np.array([-1,0,1,2,3])
C=sparse.dia_matrix((data,offsets),shape=(N+1,N+1),dtype='float64')
aa=C.todense()
# Cinv=
def energy(n):
    v=C@np.array([n]).T
    return np.sum(n*(v.flatten()/2+offset_q))

def transition_rate(n2,n1,kBT):
    
    dE=energy(n2)-energy(n1)
    
    Gamma=dE/(1-np.exp(-dE/kBT))
    return Gamma
    
