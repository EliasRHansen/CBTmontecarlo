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
#########Inpu
kB=8.617*1e-5

e_SI=1.602*1e-19

class CBTmontecarlo:
    
    def __init__(self,N,offset_q,n0,U,T,Cs,offset_C,second_order_C,Ec,Gt,gi,dtype='float64'):
        self.Ec=2*Ec #units of eV
        self.N=N
        self.n0=n0.astype(dtype) #units of number of electrons
        self.n=n0.astype(dtype)
        self.Cs=Cs.astype(dtype) #units of e/Ec=160 [fmF]/Ec[microeV]
        self.second_order_C=second_order_C.astype(dtype)#units of e/Ec=160[fmF]/Ec[microeV]
        self.U=U #units of eV
        self.offset_q=offset_q.astype(dtype) #units of number of electrons
        self.offset_C=offset_C.astype(dtype)#units of e/Ec=160[fmF]/Ec[microeV]
        self.neff=self.neff_f(self.n0)
        self.kBT=kB*T #units of eV
        self.u=self.Ec/(2*self.kBT)
        self.Gt=Gt
        self.dtype=dtype
        normalization=sum(1/gi)
        self.gi=gi.astype(dtype)*normalization
        d1=np.concatenate((self.Cs,np.zeros((1,))))
        dm1=np.concatenate((self.Cs,np.zeros((1,))))
        d2=np.concatenate((self.second_order_C[0:-1],np.zeros((2,))))
        dm2=np.concatenate((self.second_order_C[1::],np.zeros((2,))))
        self.pi=np.ones((self.N,))/N
        self.dtp=[]
        self.dQp=[]
        self.dtp2=[]
        self.dQp2=[]
        self.Ip=[]
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
        potentials=self.Cinv@np.array([self.n],dtype=dtype).T
        # potentials=np.array(potentials,dtype='float64')
        potentials=list(potentials.flatten())
        
        potentials.append(self.U/(2*self.Ec))
        potentials.append(-self.U/(2*self.Ec))
        self.potentials=np.roll(np.array(potentials,dtype=dtype),1)
        dataM=np.array([[-1]*self.N,[1]*self.N])
        offsetsM=np.array([0,1])
        self.M=sparse.dia_matrix((dataM,offsetsM),shape=(self.N-1,self.N),dtype=dtype).toarray()
    
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
    def energy(self,n,boundary_work=True,number_of_concurrent=1):
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
        if number_of_concurrent==1:
            if n.shape==(self.N-1,):
                v=self.Cinv@np.array([n]).T
                boundaries=(self.Cs[0]*v[0]+self.second_order_C[1]*v[1]-self.Cs[-1]*v[-1]-self.second_order_C[-1]*v[-2])*self.U/(2*self.Ec) #units of Ec
                E=np.sum(n*(v.flatten()/2+self.offset_q))+boundaries[0]
                return E #units of Ec
            elif n.ndim==2:
                
                v=self.Cinv@n 
                # v=np.einsum('ij,jl->il',A,n)
                w=np.einsum('ij,ij->j',n,v)     
                # ww=n.T@np.array([self.offset_q]).T
                boundaries=(self.Cs[0]*v[0,:]+self.second_order_C[1]*v[1,:]-self.Cs[-1]*v[-1,:]-self.second_order_C[-1]*v[-2,:])*self.U/(2*self.Ec) #units of Ec. Negative sign because the matrix elements have funny signs.
                E=w.flatten()/2+boundaries#+ww.flatten() #units of Ec
                if boundary_work:
                    E[::2*self.N]=E[::2*self.N]+self.U/(2*self.Ec)
                    E[self.N-1::2*self.N]=E[self.N-1::2*self.N]+self.U/(2*self.Ec)
                    E[self.N::2*self.N]=E[self.N::2*self.N]-self.U/(2*self.Ec)
                    E[2*self.N-1::2*self.N]=E[2*self.N-1::2*self.N]-self.U/(2*self.Ec)
                # if boundary_work:
                #     E[0]=E[0]+self.U/(2*self.Ec)
                #     E[N-1]=E[N-1]+self.U/(2*self.Ec)
                #     E[N]=E[N]-self.U/(2*self.Ec)
                #     E[-1]=E[-1]-self.U/(2*self.Ec)
    
                return E #units of Ec
            else:
                raise Exception('energy could not be calculated due to incorrect shape of charge array')
        else:


                
                v=self.Cinv@n 
                # v=np.einsum('ij,jl->il',A,n)
                w=np.einsum('ij,ij->j',n,v)     
                # ww=n.T@np.array([self.offset_q]).T
                boundaries=(self.Cs[0]*v[0,:]+self.second_order_C[1]*v[1,:]-self.Cs[-1]*v[-1,:]-self.second_order_C[-1]*v[-2,:])*self.U/(2*self.Ec) #units of Ec. Negative sign because the matrix elements have funny signs.
                E=w.flatten()/2+boundaries#+ww.flatten() #units of Ec

                if boundary_work:
                    E[::2*self.N]=E[::2*self.N]+self.U/(2*self.Ec)
                    E[self.N-1::2*self.N]=E[self.N-1::2*self.N]+self.U/(2*self.Ec)
                    E[self.N::2*self.N]=E[self.N::2*self.N]-self.U/(2*self.Ec)
                    E[2*self.N-1::2*self.N]=E[2*self.N-1::2*self.N]-self.U/(2*self.Ec)

                return E #units of Ec

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

    def Q(self,n,reverse=False,number_of_concurrent=1):
        """
        

        Parameters
        ----------
        n : charge array

        Returns
        -------
        Matrix with charge arrays as columns resulting from all possible transitions away from the provided charge configuration n
            
            DESCRIPTION.

        """
        if n.ndim==1:
            Qr=np.array([n],dtype=self.dtype).T.repeat(self.N,axis=1)+self.M
            Ql=np.array([n],dtype=self.dtype).T.repeat(self.N,axis=1)-self.M
            if reverse==False:
                self.Q_new=np.concatenate((Qr,Ql),axis=1)
            else:
                self.Q_new=np.concatenate((Ql,Qr),axis=1)
            return np.concatenate((Qr,Ql),axis=1)
        elif n.ndim==2:
            Qout=np.zeros((N-1,2*N,number_of_concurrent))
            Q=np.array([n],dtype=self.dtype).repeat(2*self.N,axis=-1).reshape(N-1,2*N*number_of_concurrent)+np.concatenate((self.M,-self.M),axis=1).repeat(number_of_concurrent,axis=1)
            
            return Q
    def Q0(self,n,number_of_concurrent=1):
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
        if n.ndim==1:
            Qr=np.array([n],dtype=self.dtype).T.repeat(self.N,axis=1)
            self.Q_old=np.concatenate((Qr,Qr),axis=1)
            return np.concatenate((Qr,Qr),axis=1)
        elif n.ndim==2:
            Q=np.array([n],dtype=self.dtype).repeat(2*self.N,axis=-1).reshape(N-1,2*N*number_of_concurrent)
            return Q

    def update_transition_rate(self,n2,n1,number_of_concurrent=1,update_gammas=True):
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
        limit1=1e-9
        limit2=1e9
        dE=-(self.energy(n2)-self.energy(n1,boundary_work=False)) #units of Ec
        if number_of_concurrent==1:
            


            if dE.ndim==1:
                Gamma=np.zeros_like(dE)
                Gamma[(-dE*2*self.u>np.log(limit1)) & (-dE*2*self.u<np.log(limit2))]=dE[(-dE*2*self.u>np.log(limit1)) & (-dE*2*self.u<np.log(limit2))]/(1-np.exp(-dE[(-dE*2*self.u>np.log(limit1)) & (-dE*2*self.u<np.log(limit2))]*2*self.u))
                Gamma[-dE*2*self.u<=np.log(limit1)]=dE[-dE*2*self.u<=np.log(limit1)]
                Gamma[-dE*2*self.u>=np.log(limit2)]=-dE[-dE*2*self.u>=np.log(limit2)]*np.exp(dE[-dE*2*self.u>=np.log(limit2)]*2*self.u)
                # print('updating transition rates')
                # Gamma=Gamma
                if update_gammas:
                    self.gammas=Gamma
                return Gamma.astype(self.dtype)
            elif self.iterable(dE)==False:
                
                if (-dE*2*self.u>np.log(limit1)) and (-dE*2*self.u<np.log(limit2)):
                    Gamma=dE/(1-np.exp(-dE*2*self.u))
                elif (-dE*2*self.u<=np.log(limit1)):
                    Gamma=dE
                elif (-dE*2*self.u>=np.log(limit2)):
                    Gamma=-dE*np.exp(dE*2*self.u)
                # Gamma=Gamma
                return Gamma
        else:
            Gamma=np.zeros_like(dE)
            Gamma[(-dE*2*self.u>np.log(limit1)) & (-dE*2*self.u<np.log(limit2))]=dE[(-dE*2*self.u>np.log(limit1)) & (-dE*2*self.u<np.log(limit2))]/(1-np.exp(-dE[(-dE*2*self.u>np.log(limit1)) & (-dE*2*self.u<np.log(limit2))]*2*self.u))
            Gamma[-dE*2*self.u<=np.log(limit1)]=dE[-dE*2*self.u<=np.log(limit1)]
            Gamma[-dE*2*self.u>=np.log(limit2)]=-dE[-dE*2*self.u>=np.log(limit2)]*np.exp(dE[-dE*2*self.u>=np.log(limit2)]*2*self.u)
            # print('updating transition rates')
            # Gamma=Gamma
            if update_gammas:
                self.gammas=Gamma
                self.gammas2=Gamma.reshape(number_of_concurrent,2*self.N)
            return Gamma
    
    def P(self,n,update=False,number_of_concurrent=1):
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
                self.p=number_of_concurrent*p/sum(p)
                return self.p
            except Exception:
                p=self.update_transition_rate(self.Q(n),self.Q0(n))
                self.p=number_of_concurrent*p/sum(p)
                return self.p
        else:
                p=self.update_transition_rate(self.Q(n),self.Q0(n))
                self.p=number_of_concurrent*p/sum(p)
                return self.p


    def pick_event(self,n,k,number_of_concurrent=1):
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
        if number_of_concurrent==1:
            index=random.choices(np.arange(2*self.N),weights=self.P(n),k=k)
            
            return index
        else:
            indices=[]
            Ps=self.P(n,number_of_concurrent)
            for s in np.arange(number_of_concurrent):
                
                indices.append(random.choices(np.arange(2*self.N)+2*self.N*s,weights=Ps[s*2*N:(s+1)*2*N],k=k))
            
            return indices
    
    def dt_f(self,n,number_of_concurrent=1):
        factor_SI=e_SI/(self.N*self.Ec*self.Gt)
        if number_of_concurrent==1:
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
        else:
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
        
    def dt_and_dQ_2_f(self,Qn,number_of_concurrent=1):
        factor_SI=e_SI/(self.N*self.Ec*self.Gt)
        # self.dts=factor_SI/(self.gammas[0:self.N]+self.gammas[self.N::])
        QQn=np.array([self.Q(n) for n in Qn.T])
        QQn=eo.rearrange(QQn,'i j k -> j (i k)')
        Q0Qn=np.array([self.Q0(n) for n in Qn.T])
        Q0Qn=eo.rearrange(Q0Qn,'i j k -> j (i k)')
        Gam=self.update_transition_rate(QQn,Q0Qn,update_gammas=False).reshape(2*self.N,2*self.N)
        Gamsum=np.sum(Gam,axis=1)
        dt2=factor_SI/Gamsum
        dt2=np.sum(dt2*self.p)
        
        dQ2=-(e_SI/self.N)*np.sum(Gam[:,0:self.N]-Gam[:,self.N::],axis=1)/Gamsum
        dQ2=np.sum(dQ2*self.p)
        return dt2,dQ2
    

    def dQ_f(self,n):
        
        try:
            self.dQ=-(e_SI/self.N)*sum(self.gammas[0:self.N]-self.gammas[self.N::])/sum(self.gammas)
            return self.dQ
        except Exception:
            self.update_transition_rate(self.Q(n),self.Q0(n))
            self.dQ=-(e_SI/self.N)*sum(self.gammas[0:self.N]-self.gammas[self.N::])/sum(self.gammas)
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
        Qneff=self.Q(neff)
        Q0neff=self.Q0(neff)
        self.update_transition_rate(Qneff,Q0neff)
        self.index=self.pick_event(neff,1)
        Q_new=self.Q(self.n)
        n_new=Q_new[:,self.index[0]]
        if store_data:
            self.dtp.append(self.dt_f(neff))
            self.dQp.append(self.dQ_f(neff))
            dt,dQ=self.dt_and_dQ_2_f(Qneff)
            self.dtp2.append(dt)
            self.dQp2.append(dQ)
        self.n=n_new
        if store_data:
            self.store_n()
            self.ntot_f()
    def initiate_multistep(self,number_of_concurrent,randomize_initial=True):
        
        self.ns=np.array([self.n0]*number_of_concurrent)
        if randomize_initial==True:
            for j in np.arange(number_of_concurrent-1):
                
                self.ns[j+1]=self.n0+np.array(random.choices(np.arange(11)-5,k=N-1,weights=np.exp(-0.2*(np.arange(11)-5)**2)))
        
        return self.ns.T
    def multistep(self,store_data=False,number_of_concurrent=1):
        neff=self.neff_fnD(self.ns)
        Qneff=self.Q(neff)
        Q0neff=self.Q0(neff)
        self.update_transition_rate(Qneff,Q0neff)
        if store_data:
            dt,dQ=self.dt_and_dQ_2_f(Qneff)
            self.dtp.append(self.dt_f(neff))
            self.dQp.append(self.dQ_f(neff))
            self.dtp2.append(dt)
            self.dQp2.append(dQ)
            # self.Ip.append(np.sum(np.array(self.dQp))/np.sum(np.array(self.dtp)))
        self.indices=self.pick_event(neff,1,number_of_concurrent)
        Q_new=self.Q(self.ns)
        n_new=Q_new[:,self.indices]
        self.n=n_new
        if store_data:
            self.store_n()
            self.ntot_f()

    
    def __call__(self,number_of_steps=1,transient=0,print_every=None,number_of_concurrent=1):
        if number_of_concurrent==1:
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
            self.final_current=np.sum(np.array(self.dQp)[transient::])/np.sum(np.array(self.dtp)[transient::])
            return self.final_current
        else:
            print('not implemented')
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
    Cs=np.ones((N,))
    offset_C=-0*np.ones((N-1,))*1e-4
    offset_q=0*n0/N
    second_order_C=0*Cs*1e-1
    Ec=4e-6
    Gt=2e-5
    gi=np.ones((2*N,))

    T=0.015
    FWHM=5.439*kB*T*N
    
    points=21
    lim=3.5*FWHM
    dV=FWHM/100
    # Us=np.linspace(-lim,lim,points)
    # Us=np.concatenate((Us,np.linspace(-lim,lim,points)+5e-6))
    # Us=np.concatenate((Us,np.linspace(-lim,lim,points)-5e-6))
    Us=np.linspace(-lim,lim,points)-dV
    Us=np.concatenate((Us,np.linspace(-lim,lim,points)))
    Us=np.concatenate((Us,np.linspace(-lim,lim,points)+dV))
    U=Us[2]
    # currentss=[]
    a=time()
    CBT=CBTmontecarlo(N,offset_q,n0,U,T,Cs,offset_C,second_order_C,Ec,Gt,gi,dtype='float64')
    number_of_steps=50000
    transient=2
    current=CBT(number_of_steps,transient,print_every=1000)
    
    b=time()
    print(b-a)
    a=time()
    def f(U):
        
        CBT=CBTmontecarlo(N,offset_q,n0,U,T,Cs,offset_C,second_order_C,Ec,Gt,gi,dtype='float64')
        number_of_steps=50000
        transient=2
        print_every=1000
        current=CBT(number_of_steps,transient,print_every=print_every)
        dQ=np.array(CBT.dQp)[transient::]
        dt=np.array(CBT.dtp)[transient::]
        sigI=np.sqrt(np.var(dQ)/np.sum(np.array(dt))**2+(np.sum(dQ)/np.sum(dt))**2*np.var(dt)/np.sum(dt)**2)*np.sqrt(number_of_steps/print_every-transient)
        
        current2=np.sum(np.array(CBT.dQp2)[transient::])/np.sum(np.array(CBT.dtp2)[transient::])
        dQ=np.array(CBT.dQp2)[transient::]
        dt=np.array(CBT.dtp2)[transient::]
        sigI2=np.sqrt(np.var(dQ)/np.sum(np.array(dt))**2+(np.sum(dQ)/np.sum(dt))**2*np.var(dt)/np.sum(dt)**2)*np.sqrt(number_of_steps/print_every-transient)
        return current,current2,sigI,sigI2

    
    Is=Parallel(n_jobs=8,verbose=50)(delayed(f)(U) for U in Us)
    current=np.array([I[0] for I in Is])

    
    gm1=(current[2*points:3*points]-current[0:points])/(Us[2*points:3*points]-Us[0:points])
    

    
    def CBT_model_g(x):
        return (x*np.sinh(x)-4*np.sinh(x/2)**2)/(8*np.sinh(x/2)**4)



    def CBT_model_G(V):
        
        return Gt*(1-Ec*CBT_model_g(V/(N*kB*T))/(kB*T))
    
    plt.figure()
    plt.plot(Us[points:2*points],gm1,'.')
    plt.plot(np.linspace(Us[0],Us[-1],1000),CBT_model_G(np.linspace(Us[0],Us[-1],1000)))
    

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