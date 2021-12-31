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
#########Inpu
kB=8.617*1e-5
e_SI=1.602*1e-19
class CBTmontecarlo:
    
    def __init__(self,N,offset_q,n0,U,T,Cs,offset_C,second_order_C,Ec,Gt,gi):
        self.Ec=Ec #units of eV
        self.N=N
        self.n0=n0 #units of number of electrons
        self.n=n0
        self.Cs=Cs #units of e/Ec=160 [fmF]/Ec[microeV]
        self.second_order_C=second_order_C#units of e/Ec=160[fmF]/Ec[microeV]
        self.U=U #units of eV
        self.offset_q=offset_q #units of number of electrons
        self.offset_C=offset_C#units of e/Ec=160[fmF]/Ec[microeV]
        self.neff=self.neff_f(self.n0)
        self.kBT=kB*T #units of eV
        self.u=self.Ec/self.kBT
        self.Gt=Gt
        normalization=sum(1/gi)
        self.gi=gi*normalization
        d1=np.concatenate((self.Cs,np.zeros((1,))))
        dm1=np.concatenate((self.Cs,np.zeros((1,))))
        d2=np.concatenate((self.second_order_C[0:-1],np.zeros((2,))))
        dm2=np.concatenate((self.second_order_C[1::],np.zeros((2,))))
        self.dtp=[]
        self.dQp=[]
        dm1=np.roll(dm1,-1)
        dm2=np.roll(dm2,-1)
        self.ntot=[sum(self.n0)]
        self.n_history=[self.n0]
        d0=np.roll(dm2,2)+np.roll(dm1,1)+np.roll(d1,-1)+np.roll(d2,-2)+np.concatenate((self.offset_C,np.zeros((2,))))
        
        data=np.array([-dm2,-dm1,d0,-d1,-d2])
        data=data[:,0:-2]
        offsets=np.array([-2,-1,0,1,2])
        self.C=sparse.dia_matrix((data,offsets),shape=(N-1,N-1),dtype='float64')
        self.Cinv=inv(sparse.csc_matrix(self.C))
        potentials=self.Cinv@np.array([self.n]).T
        # potentials=np.array(potentials,dtype='float64')
        potentials=list(potentials.flatten())
        
        potentials.append(self.U/(2*self.Ec))
        potentials.append(-self.U/(2*self.Ec))
        self.potentials=np.roll(np.array(potentials),1)
        dataM=np.array([[-1]*self.N,[1]*self.N])
        offsetsM=np.array([0,1])
        self.M=sparse.dia_matrix((dataM,offsetsM),shape=(self.N-1,self.N),dtype='float64').toarray()
    
    def neff_f(self,n):
        """
        

        Parameters
        ----------
        n : charge array
            DESCRIPTION.

        Returns
        -------
        charge array adjusted for the fact that the potential at the boundaries are not otherwise included.
            DESCRIPTION.

        """
        self.neff=copy(n)
        self.neff[0]=self.neff[0]-self.Cs[0]*self.U/2
        self.neff[1]=self.neff[1]-self.second_order_C[1]*self.U/2
        self.neff[-1]=self.neff[-1]+self.Cs[-1]*self.U/2
        self.neff[-2]=self.neff[-2]+self.second_order_C[-1]*self.U/2
        return self.neff
    def neff_finv(self,neff):
        """
        

        Parameters
        ----------
        n : charge array
            DESCRIPTION.

        Returns
        -------
        charge array adjusted for the fact that the potential at the boundaries are not otherwise included.
            DESCRIPTION.

        """
        n=copy(neff)
        n[0]=n[0]+self.Cs[0]*self.U/2
        n[1]=n[1]+self.second_order_C[1]*self.U/2
        n[-1]=n[-1]-self.Cs[-1]*self.U/2
        n[-2]=n[-2]-self.second_order_C[-1]*self.U/2
        return n
    def neff_fnD(self,n):
        """
        

        Parameters
        ----------
        n : charge array
            DESCRIPTION.

        Returns
        -------
        charge array adjusted for the fact that the potential at the boundaries are not otherwise included.
            DESCRIPTION.

        """
        self.neffnD=copy(n)
        self.neffnD[0,:]=self.neffnD[0,:]-self.Cs[0]*self.U/2
        self.neffnD[1,:]=self.neffnD[1,:]-self.second_order_C[1]*self.U/2
        self.neffnD[-1,:]=self.neffnD[-1,:]+self.Cs[-1]*self.U/2
        self.neffnD[-2,:]=self.neffnD[-2,:]+self.second_order_C[-1]*self.U/2
        return self.neffnD
    def energy(self,n,boundary_work=True):
        """
        

        Parameters
        ----------
        n : charge array, or matrix of 2N charge arrays as columns
            DESCRIPTION.

        Raises
        ------
        Exception
            If the charge array doesnt have the right shape.

        Returns
        -------
        Total energy in the system, or array of total energies if the input is a matrix with charge arrays as columns. The units are Ec.
            DESCRIPTION.

        """
        if n.shape==(self.N-1,):
            v=self.Cinv@np.array([n]).T
            boundaries=(self.Cs[0]*v[0]+self.second_order_C[1]*v[1]-self.Cs[-1]*v[-1]-self.second_order_C[-1]*v[-2])*self.U/(2*self.Ec) #units of Ec
            E=np.sum(n*(v.flatten()/2+self.offset_q))+boundaries[0]
            return E #units of Ec
        elif n.shape==(self.N-1,2*self.N):
            
            v=self.Cinv@n
            w=np.einsum('ij,ij->j',n,v)     
            # ww=n.T@np.array([self.offset_q]).T
            boundaries=(self.Cs[0]*v[0,:]+self.second_order_C[1]*v[1,:]-self.Cs[-1]*v[-1,:]-self.second_order_C[-1]*v[-2,:])*self.U/(2*self.Ec) #units of Ec. Negative sign because the matrix elements have funny signs.
            E=w.flatten()/2+boundaries#+ww.flatten() #units of Ec

            if boundary_work:
                E[0]=E[0]+self.U/(2*self.Ec)
                E[N-1]=E[N-1]+self.U/(2*self.Ec)
                E[N]=E[N]-self.U/(2*self.Ec)
                E[-1]=E[-1]-self.U/(2*self.Ec)

            return E #units of Ec
        else:
            raise Exception('energy could not be calculated due to incorrect shape of charge array')
    # def energy(self,n):
    #     """
        

    #     Parameters
    #     ----------
    #     n : charge array, or matrix of 2N charge arrays as columns
    #         DESCRIPTION.

    #     Raises
    #     ------
    #     Exception
    #         If the charge array doesnt have the right shape.

    #     Returns
    #     -------
    #     Total energy in the system, or array of total energies if the input is a matrix with charge arrays as columns. The units are Ec.
    #         DESCRIPTION.

    #     """
    #     if n.shape==(self.N-1,):
    #         v=self.Cinv@np.array([n]).T

    #         Dphi=np.diff(v.flatten())
            
    #         return np.sum(Cs[1:-1]*(Dphi)**2/2)+Cs[0]*(v[0]+U/(2*Ec))**2/2+Cs[-1]*(v[-1]-U/(2*Ec))**2/2 #units of Ec
    #     elif n.shape==(self.N-1,2*self.N):
    #         v=self.Cinv@n

    #         Dphi=v-np.roll(v,1,axis=0)
    #         Dphi=Dphi[1::,:]**2
    #         Dphi0=(v[0,:]+U/(2*Ec))**2
    #         Dphi1=(v[-1,:]-U/(2*Ec))**2
    #         w=np.einsum('i,ij->j',Cs[1:-1],Dphi)+ Cs[0]*Dphi0+Cs[-1]*Dphi1
    #         #boundaries=(self.Cs[0]*v[0,:]+self.second_order_C[1]*v[1,:]-self.Cs[-1]*v[-1,:]-self.second_order_C[-1]*v[-2,:])*self.U/(2*self.Ec) #units of Ec
    #         return w.flatten()/2 #units of Ec
    #     else:
    #         raise Exception('energy could not be calculated due to incorrect shape of charge array')
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

    def Q(self,n,reverse=False):
        """
        

        Parameters
        ----------
        n : charge array

        Returns
        -------
        Matrix with charge arrays as columns resulting from all possible transitions away from the provided charge configuration n
            
            DESCRIPTION.

        """
        
        Qr=np.array([n]).T.repeat(self.N,axis=1)+self.M
        Ql=np.array([n]).T.repeat(self.N,axis=1)-self.M
        if reverse==False:
            self.Q_new=np.concatenate((Qr,Ql),axis=1)
        else:
            self.Q_new=np.concatenate((Ql,Qr),axis=1)
        return np.concatenate((Qr,Ql),axis=1)
    def Q0(self,n):
        """
        

        Parameters
        ----------
        n : charge array
            DESCRIPTION.

        Returns
        -------
        Matrix with copies of current charge array as columns
        
            DESCRIPTION.

        """
        Qr=np.array([n]).T.repeat(self.N,axis=1)
        self.Q_old=np.concatenate((Qr,Qr),axis=1)
        return np.concatenate((Qr,Qr),axis=1)

    def update_transition_rate(self,n2,n1):
        """
        

        Parameters
        ----------
        n2 : charge array or matrix of charge arrays as columns
            DESCRIPTION.
        n1 : charge array or matrix of charge arrays as columns
            DESCRIPTION.

        Returns
        -------
        Gamma : array of transition rates calculated from the energy differences between charge configuration n2 and n1
            DESCRIPTION.

        """
        dE=-(self.energy(n2)-self.energy(n1,boundary_work=False)) #units of Ec
        limit1=1e-15
        limit2=1e15
        # Gamma=dE/(1-np.exp(-dE*self.u))
        # self.gammas=Gamma
        # return Gamma
        if dE.shape==(2*self.N,):
            Gamma=np.zeros_like(dE)
            Gamma[(-dE*self.u>np.log(limit1)) & (-dE*self.u<np.log(limit2))]=dE[(-dE*self.u>np.log(limit1)) & (-dE*self.u<np.log(limit2))]/(1-np.exp(-dE[(-dE*self.u>np.log(limit1)) & (-dE*self.u<np.log(limit2))]*self.u))
            Gamma[-dE*self.u<=np.log(limit1)]=dE[-dE*self.u<=np.log(limit1)]
            Gamma[-dE*self.u>=np.log(limit2)]=-dE[-dE*self.u>=np.log(limit2)]*np.exp(dE[-dE*self.u>=np.log(limit2)]*self.u)
            # print('updating transition rates')
            # Gamma=Gamma
            self.gammas=Gamma
            return Gamma
        elif self.iterable(dE)==False:
            
            if (-dE*self.u>np.log(limit1)) and (-dE*self.u<np.log(limit2)):
                Gamma=dE/(1-np.exp(-dE*self.u))
            elif (-dE*self.u<=np.log(limit1)):
                Gamma=dE
            elif (-dE*self.u>=np.log(limit2)):
                Gamma=-dE*np.exp(dE*self.u)
            # Gamma=Gamma
            return Gamma
    
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
        if update==False:
            try:
                p=self.gammas
                return p/sum(p)
            except Exception:
                p=self.update_transition_rate(self.Q(n),self.Q0(n))
                return p/sum(p)
        else:
                p=self.update_transition_rate(self.Q(n),self.Q0(n))
                return p/sum(p)


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
        index=random.choices(np.arange(2*self.N),weights=self.P(n),k=k)
        
        return index
    
    def dt_f(self,n):
        factor_SI=e_SI/(2*self.N*self.Ec*self.Gt)
        try:
            # self.dts=factor_SI/(self.gammas[0:self.N]+self.gammas[self.N::])
            self.dts=factor_SI/sum(self.gammas)
            return self.dts
            # return sum(self.dts)
        except Exception:
            self.update_transition_rate(self.Q(n),self.Q0(n))
            # self.dts=factor_SI/(self.gammas[0:self.N]+self.gammas[self.N::])
            return self.dts
            self.dts=factor_SI/sum(self.gammas)
            # return sum(self.dts)

    def dQ_f(self,n):
        try:
            self.dQ=-e_SI*sum(self.gammas[0:self.N]-self.gammas[self.N::])/sum(self.gammas)
            return self.dQ
        except Exception:
            self.update_transition_rate(self.Q(n),self.Q0(n))
            self.dQ=-e_SI*sum(self.gammas[0:self.N]-self.gammas[self.N::])/sum(self.gammas)
            return self.dQ
    def ntot_f(self):
        self.ntot.append(sum(self.n))
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
        neff=self.neff_f(self.n)
        self.update_transition_rate(self.Q(neff),self.Q0(neff))
        if store_data:
            self.dtp.append(self.dt_f(neff))
            self.dQp.append(self.dQ_f(neff))
        self.index=self.pick_event(neff,1)
        Q_new=self.Q(self.n)
        n_new=Q_new[:,self.index[0]]
        self.n=n_new
        if store_data:
            self.store_n()
            self.ntot_f()

    
    def __call__(self,number_of_steps=1,transient=0,print_every=None):
        if print_every is None:
            print_every=int(number_of_steps/100+1)
        # fig2,ax2=plt.subplots()
        for i in np.arange(number_of_steps):
            if print_every != 0:
                if i%print_every==0:
                    print('{:.1f}'.format(i*100/number_of_steps)+' pct.')

                    self.step(store_data=True)
                else:
                    self.step(store_data=False)
            else:
                self.step(store_data=False)
        final_current=np.array(self.dQp)[transient::]/np.array(self.dtp)[transient::]
        return final_current
        
    # def initialize_many(self,n,k):
    #     self.ns=np.array([n]*k)
    #     self.dtps=[[]]*k
    #     self.dQs=[[]]*k
        
    # def step_many(self,k):
        
    #     neffnD=self.neff_fnD(self.ns)
    #     Q_new=[]
    #     for j in np.arange(k):
    #         self.update_transition_rate(self.Q(neffnD[j]),self.Q0(neffnD[j]))
    #         self.dtps[j].append(self.dt_f(neffnD[j]))
    #         self.dQps[j].append(self.dQ_f(neffnD[j]))
    #         Q_new.append(self.Q(self.ns[j]))
    #     self.index=self.pick_event(neff,k)

    #     n_new=Q_new[:,self.index[0]]
    #     self.n=n_new
        
if __name__=='__main__':
    
    N=100
    test=np.linspace(1,N,N)

    n0=0*np.array(random.choices(np.arange(11)-5,k=N-1,weights=np.exp(-0.2*(np.arange(11)-5)**2)))#-np.ones((N-1,))*10
    # n0[40:60]=n0[40:60]+100
    Cs=np.ones((N,))/2
    offset_C=-0*np.ones((N-1,))*1e-4
    offset_q=0*n0/N
    second_order_C=0*Cs*1e-1
    Ec=1e-6
    Gt=2e-5
    gi=np.ones((2*N,))
    U=-5e-3
    T=0.03
    
    
    points=40
    lim=6e-3
    Us=np.linspace(-lim,lim,points)
    # Us=np.concatenate((Us,np.linspace(-lim,lim,points)+5e-6))
    # Us=np.concatenate((Us,np.linspace(-lim,lim,points)-5e-6))
    Us=np.concatenate((Us,np.linspace(-lim,lim,points)-15e-6))
    Us=np.concatenate((Us,np.linspace(-lim,lim,points)+15e-6))
    currentss=[]
    a=time()
    # def f(number_of_steps,transient):
    #     CBT=
    fig1,ax1=plt.subplots()
    for U in Us:
        print(U)
        CBT=CBTmontecarlo(N,offset_q,n0,U,T,Cs,offset_C,second_order_C,Ec,Gt,gi)
        number_of_steps=10000
        transient=0
        currents=CBT(number_of_steps,transient,print_every=100)
        # ax1.plot(currents)
        # ax1.set_ylabel('current')
        currentss.append(currents)
        # second_order_C=Cs*1e-2
    currents=np.array(currentss)
    currentm=[]
    currentstd=[]
    for c in np.arange(len(currentss)):
        currentm.append(np.mean(currentss[c][5::]))
        currentstd.append(np.std(currentss[c][5::]))
    currentm=np.array(currentm)
    currentstd=np.array(currentstd)
    
    
    
    
    
    
    
    
    
    
    plt.figure()
    plt.errorbar(Us,currentm,yerr=currentstd,fmt='.',label='data')
    plt.legend()
    plt.ylabel('current')
    plt.xlabel('bias voltage')
    plt.figure()
    Us=np.linspace(-10e-3,10e-3,1000)
    G3 = dxdt(currentm[::20],Us[::20], kind="finite_difference", k=2)
    plt.errorbar(Us[0:-1],np.diff(currentm)/np.diff(Us),fmt='.',label='data')
    plt.errorbar(Us[::20],G3,fmt='.',label='data')
    
    plt.legend()
    plt.ylabel('G')
    plt.xlabel('bias voltage')
    b=time()
    # processed_list = Parallel(n_jobs=8,verbose=50)(delayed(f)(j,self) for j in np.arange(int(len(self.z_SI))))
    print(b-a)
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