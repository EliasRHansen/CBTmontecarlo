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
from scipy.special import exprel
from itertools import product
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
np.seterr(all = 'raise')
kB=8.617*1e-5
e_SI=1.602*1e-19


def iterable(m):
    """


    Parameters
    ----------
    m : any type
        Object to be checked for iterability.

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
    """


    Parameters
    ----------
    x : 2d array of floats
        1st dimension is probabilities for each transition of the particular charge array represented by the zeroth dimension

    Returns
    -------
    index : array of int
        index of transition event for each charge array chosen according to the probabilities provided.

    """
    i=x.shape[0]
    index=np.zeros((i),dtype='int32')
    for P in np.arange(i):
        index[P]=np.searchsorted(np.cumsum(x[P]), np.random.random(), side="right")
    return index
def split_voltages(V,dV):
    """


    Parameters
    ----------
    V : 1d array of float
        voltages corresponding to conductances.
    dV : float
        size of voltage differential used to compute conductance from currents.

    Returns
    -------
    Us : 1d array of twice the length of V
        If V is the desired voltages to find the corresponding conductances, the simulation will run for each value of Us to find the current.

    """
    Us=V-dV
    Us=np.concatenate((Us,V+dV))
    return Us

class CBTmain: #just does the simulation, no further analysis


    def __init__(self,U,T,Ec,Gt,N,Nruns,Ntransient,number_of_concurrent,Ninterval,skip_transient,parallelization='external',
                 n0=None,second_order_C=None,dtype='float64',offset_C=None,dC=0,n_jobs=2,batchsize=1,q0=0):
        """


        Parameters
        ----------
        U : 1d array of float, or float
            Voltages.
        T : float
            temperature in Kelvin.
        Ec : float
            charging energy. Only the ratio of the charging energy to the the temperature matters for the actual shape.
            At fixed ratio, changing one the the parameters amounts to changing the horizotal scaling.
        Gt : float
            tunneling conductance.
        N : int
            number of islands.
        Nruns : int
            total number of steps to be taken by each concurrent simulation (after transient steps are taken).
        Ntransient : int
            number of transient steps.
        number_of_concurrent : int
            number of data points to generate for each voltage (to save time charge configurations are "launched" from the same initial state reached after the transient regime; however, the initial state will be different for each voltage, since the probable steady state configurations will be different).
        Ninterval : int
            interval between datapoint storage of charge differentials and time differentials.
            decreasing Ninterval obviously increases simulation time (albeit very little),
            and decreasing is below the autocorrelation time doesnt decrease the variance of the final data
            (however, the covariance of the current/conductance data points are not increased by decreasing
             Ninterval since those are generated from seperate parallel runs).
        skip_transient : bool
            should be true. "skips" the data from the transient regime. Only reason to set it false would be to see how the current evolve in time from the highly improbable initial state.
        parallelization : str, optional
            "external": simulations for each voltage are run in parallel using the joblib library in batches of size given by the input batchsize.
                        Debugging can be very hard 'external' since references are made to the internals of the joblib module rather than the code.
            "internal": simulations for each voltage are run in parallel by fancy numpy vectorization, but uses a for-loop to iiterate over batches.
            "non": for loop structure is used to iterate over voltages.

            The only reason not to use "external" is if the jobliib library somehow fails or interferes with other code.

            The default is 'external'.
        n0 : 1D-array of float of size N-1, optional
            initial charge configuration. The default is zeros.
        second_order_C : array of float, optional
            coupling between next to nearest neighbours. The default is zeros.
        dtype : str, optional
            datatype of the arrays used in calculations; using float32 should be faster, but it raises a lot of floatingpointerrors, so it shouldnt be touched I think. The default is 'float64'.
        offset_C : 1D array of float, optional
            capacitances representing coupling of dots to external charges. The units are e/Ec. The default is just zeros.
        dC : float, or 1D array, optional
            variations in capacitances in units of e/Ec. The default is 0.
        n_jobs : int, optional
            number of processes to run in parallel by the joblib module. The default is 2.
        batchsize : int, optional
            number of processes to run in parallel using 'internal' numpy vectorization. The default is 1.
        q0 : int, optional
            charge offset in units of e. The default is 0.

        Returns
        -------
        CBT object
            The raw result of the simulation to be inserted into a CBT_data_analysis instance to extranct currents and conductances.

        """
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
        self.dC=dC
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

        self.Ec_factor=2/np.mean(d0)
        data=np.array([-dm2,-dm1,d0,-d1,-d2])*self.Ec_factor
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
        total=number_of_concurrent*self.number_of_Us

        print('The total number of simulations (that gives rise to a datapoint for current) is: '+str(total))
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
                print('the total number of tasks that will run be run by the joblib module is ' +str(len(voltbatch)))
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
                print('the total number of tasks is ' +str(len(voltages)))
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
        """


        Parameters
        ----------
        number_of_concurrent : int
            number of simulations to run in parallel for each voltage.

        Returns
        -------
        None.

        """
        self.number_of_concurrent=number_of_concurrent
        self.MMM_withoutU=np.tile(self.MM,(1,number_of_concurrent))
        self.MMM=np.tile(self.MM,(1,number_of_concurrent*self.number_of_Us))
        self.B_withoutU=self.Cinv@self.MMM_withoutU
        self.BB=self.Cinv@self.MM
        self.BB=np.array(self.BB,dtype=self.dtype)
        if iterable(self.U)==False:
            C=np.einsum('ij,ij->j',self.MMM_withoutU,self.B_withoutU)/2

            self.dE0=C+self.U/(2*self.Ec)*(self.Cs[0]*self.B_withoutU[0,:]-self.Cs[-1]*self.B_withoutU[-1,:]+self.second_order_C[1]*self.B_withoutU[1,:]-self.second_order_C[-1]*self.B_withoutU[-2,:])*self.Ec_factor
            self.boundary_works=self.U/(2*self.Ec)
            self.gi=np.tile(self.Cs,int(len(self.dE0)/self.N))
            # self.gi2=self.gi.reshape(self.number_of_concurrent,2*self.N)
        else:

            self.B=self.Cinv@self.MMM
            C=np.einsum('ij,ij->j',self.MMM,self.B)/2
            bound=np.kron(self.U/(2*self.Ec),(self.Cs[0]*self.B_withoutU[0,:]-self.Cs[-1]*self.B_withoutU[-1,:]+self.second_order_C[1]*self.B_withoutU[1,:]-self.second_order_C[-1]*self.B_withoutU[-2,:])*self.Ec_factor)
            self.boundary_works=self.U.repeat(number_of_concurrent)/(2*self.Ec)
            self.gi=np.tile(self.Cs,int(len(bound)/self.N))
            # self.gi2=self.gi.reshape(self.number_of_concurrent*self.number_of_Us,2*self.N)
            print('gi:'+str(self.gi))
            # print(bound/C)
            self.dE0=C+bound
        # else:
        #     print(str(self.parallelization)+' is not implemented as a value for the parameter parallelization')

    def dE_f(self,nn):
        """


        Parameters
        ----------
        nn : 2d-numpy array
            first index is the charge on the ith island, second index is a seperate charge configuration that eveolves independently.

        Returns
        -------
        E : 1d array of size 2N times the nummber of charge configurations being run in parallel
            energy differences corresponding to each possible transition of each charge array.

        """
        v=(nn.T@self.BB).flatten()
        # print(v.dtype)
        E=v+self.dE0#units of Ec
        E[::2*self.N]=E[::2*self.N]+self.boundary_works
        E[self.N-1::2*self.N]=E[self.N-1::2*self.N]+self.boundary_works
        E[self.N::2*self.N]=E[self.N::2*self.N]-self.boundary_works
        E[2*self.N-1::2*self.N]=E[2*self.N-1::2*self.N]-self.boundary_works
        # print('0:'+str(self.dE0[0]/E[0]))
        # print(self.dE0[34]/E[34])
        # print(np.mean(self.dE0/E))
        return E
    def E0(self,nn):
        v=self.Cinv@nn
        E=np.einsum('ij,ij->j',nn,v)
        return E
    # def E1(self,nn):
    #     v=self.Cinv@nn
    #     b=np.kron(self.U/(2*self.Ec),(self.Cs[0]*v[0,:]-self.Cs[-1]*v[-1,:])*self.Ec_factor)
        
    #     E=(nn.T@v).flatten()
    #     return E+b
    
    def update_transition_rate(self,n1):
        """


        Parameters
        ----------
        n1 : 2d-numpy array
            first index is the charge on the ith island, second index is a seperate charge configuration that eveolves independently.


        Returns
        -------
        Gamma : 1d array of size 2N times the nummber of charge configurations being run in parallel
            transition rates corresponding to energy differences corresponding to each possible transition of each charge array.

        """



        dE=-self.dE_f(n1)


        if dE.ndim==1:
            try:
                #Gamma=-dE/np.expm1(-dE*self.u)#(1-np.exp(-dE*self.u))
                Gamma=1/(self.u*exprel(-self.u*dE))#-dE/np.expm1(-dE*self.u)#(1-np.exp(-dE*self.u))
            except FloatingPointError:
                print('FFfloat!!')
                try:
                    Gamma=np.array(self.gammas)#np.zeros_like(dE)
                except AttributeError:
                    Gamma=np.empty(dE)
                limit1=1e-12
                limit2=1e12
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


            self.gammas=self.gi*Gamma
                # self.sumgammas=sum(self.gammas)
            self.gammas3=Gamma.reshape(self.number_of_concurrent*self.number_of_Us,2*self.N)
            return self.gammas

        # else:
        #     print('something is wrong with the dimensions of energy difference')
    def P(self,n,update=False):
        """


        Parameters
        ----------
        n : 2d-numpy array
            first index is the charge on the ith island, second index is a seperate charge configuration that eveolves independently.


        Returns
        -------
        array of 2N times M transition probabilities where the first N represents a transition from Right to left and the second N represents transitions from left to right, and M represents the number of simulations being run in parallel ('internally').


        """
        try:

            p=self.gammas2
        except Exception:
            p=self.update_transition_rate(n).reshape(self.number_of_concurrent*self.number_of_Us,2*self.N)

        self.p=np.einsum('ij,i->ij',p,1/self.Gamsum)

        return self.p

    def dt_f(self):
        """


        Returns
        -------
        1D Array of M floats representing the average time spent in each of the M present charge configurations.


        """

        self.dts=self.factor_SI/self.Gamsum
        return self.dts
    def dQ_f(self):
        """


        Returns
        -------
        1D Array of M floats representing the average charge transfer in each of the M present charge configurations.

        """

        self.dQ=-(e_SI/self.N)*self.Gamdif/self.Gamsum
        return self.dQ

    def multistep(self,store_data=False):
        """

        Take one monte carlo step for each parallel charge configuation.

        Parameters
        ----------
        store_data : bool, optional
            whether or not to store the time and charge data for the current charge configuration. The default is False.

        Returns
        -------
        None.

        """

        try:
            neff=self.ns

            self.gammas2=self.update_transition_rate(neff).reshape(self.number_of_concurrent*self.number_of_Us,2*self.N)
            self.Gamsum=np.sum(self.gammas2,axis=1)
            self.P(neff)
            if store_data:
                self.Gamdif=np.sum(self.gammas3[:,0:self.N]-self.gammas3[:,self.N::],axis=1)
                self.dtp.append(self.dt_f())
                self.dQp.append(self.dQ_f())
                # print('store E')
                # self.Ep.append(self.E0(self.ns))
            self.indices=pick_event2(self.p)#self.pick_event(neff,1)
            # self.indices=list(self.indices)
            n_new=neff+self.MM[:,self.indices]
            self.ns=n_new

        except FloatingPointError:
            print('FloatingPointError Occurred; trying to redo step. This may occur when the sum of the transition rates is very small. In this case, the sums are: '+str(self.Gamsum)+". However, I think the bug causing this is gone now.")
            print(np.sum(self.p,axis=1))
            self.indices=pick_event2(self.p)#self.pick_event(neff,1)
            # self.indices=list(self.indices)
            n_new=neff+self.MM[:,self.indices]#[:,:,0]
            self.ns=n_new
            self.multistep()

    def __call__(self,number_of_steps=1,transient=0,print_every=None,number_of_concurrent=None,skip_transient=True):
        """


        Parameters
        ----------
        number_of_steps : int, optional
            Total number of montecarlo steps (excluding transient). The default is 1.
        transient : int, optional
            number of transient steps before data are stored. The default is 0.
        print_every : int, optional
            'print_every' as in print/save data for every x step. The default is every 1/100 of the total number of steps.
        number_of_concurrent : int, optional
            number of charge configurations to run in parallel. The default is None.
        skip_transient : bool, optional
            whether or not to skip data storing in the transient regime (i.e. if False, no data points are skipped and
            all charge configurations are launched from the charge confiiguration provided as input). The default is True.

        Returns
        -------
        runs the simulation. Depending on the parallelization scheme, the main results in the form of charge
        transfer and trnsition times are either stored in the attribuute Is, or dQp and dtp together;
        this is handled automatically by inserting the resulting object into an instance of CBT_data_analysis.

        """
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
        self.Ep=[]

        if skip_transient:
            print('n0 is being moved forward through the transient window')

            self.update_number_of_concurrent(1)

            self.ns=np.array([self.n0]*self.number_of_Us).T
            print('initiating multistep for the transient window for '+str(self.number_of_concurrent)+' charge configurations to move through the transient regime')
            for j in np.arange(transient*print_every):
                if j%print_every==print_every-1:
                    print('transient {:.1f}'.format(j*100/(transient*print_every))+' pct.')
                self.multistep(store_data=False)

            # self.n0=np.array(self.n)

            del self.gammas
            del self.p
        else:
            self.ns=np.array([self.n0]*self.number_of_Us).T
        self.update_number_of_concurrent(number_of_concurrent)
        # print(self.ns)
        self.ns0=np.array(self.ns)
        self.ns=self.ns.repeat(self.number_of_concurrent,axis=1)
        print('initiating multistep for the transient window for '+str(self.number_of_concurrent*self.number_of_Us)+' charge configurations')
        # print('initial charge configuration (columns):')
        
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
        """
        This class processes the raw data. The main results are the attributes
        Gsm: the mean conductance for each voltage
        Gstd: the standard deviation of the conductance at each voltage
        Gs: The conductance data points



        Parameters
        ----------
        CBTmain_instance : instance of CBTmain class
            this contains the on processed data.
        transient : int, optional
            provides the opption of skipping some data in the beginning if the transient regime is short or skip_transient is not true in the CBTmain instance. The default is 0.

        Returns
        -------
        None.


        """
        CBT=CBTmain_instance #shorthandname
        self.raw_data=CBT
        self.now=CBT.now
        self.simulation_time=CBT.simulation_time
        self.filepath=os.getcwd()+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)
        self.q0=CBT.q0


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
                for j in np.arange(len(CBT.dQps)):
                    i1=CBT.dQps[j]
                    i2=CBT.dtps[j]
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
        """


        Parameters
        ----------
        x : floats
            unitsless variable.

        Returns
        -------
        function used in CBT model


        """
        return (x*np.sinh(x)-4*np.sinh(x/2)**2)/(8*np.sinh(x/2)**4)

    def CBT_model_G(self,V):
        """


        Parameters
        ----------
        V : float
            Voltage.

        Returns
        -------
        First order CBT conductance model. returns the conductance in units of 1/ohm


        """
        return self.raw_data.Gt*(1-self.raw_data.Ec*self.CBT_model_g(V/(self.raw_data.N*kB*self.raw_data.T))/(kB*self.raw_data.T))
    def plotG(self,save=False):
        """


        Parameters
        ----------
        save : bool, optional
            If save is true a folder will be generated in which the plots are stored. The default is False.

        Returns
        -------
        None.

        """
        points=self.points
        fig,ax=plt.subplots(figsize=(9,6))
        if iterable(self.q0):
            plt.title('MC for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={}, steps between samples={}, \n steps/(run'.format(self.raw_data.N,self.raw_data.T*1e3,self.raw_data.Ec*1e6,self.raw_data.Gt*1e6,'varying',
                                                                                                                                                      self.raw_data.Ninterval)+r'$\times$'+'datapoint)={}, runs/datapoint={}, transient interval={}'.format(self.raw_data.Nruns,self.raw_data.number_of_concurrent,self.raw_data.Ntransient))

        else:

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
        if iterable(self.q0):
            plt.title('Results for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={}'.format(self.raw_data.N,self.raw_data.T*1e3,self.raw_data.Ec*1e6,self.raw_data.Gt*1e6,'varying'))
        else:
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
        """


        Parameters
        ----------
        save : bool, optional
            If true, a folder will be created in which the plots will be saved. The default is False.

        Returns
        -------
        None.

        """
        fig,ax=plt.subplots(figsize=(9,6))
        if iterable(self.q0):
            plt.title('MC for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={}, steps between samples={}, \n steps/(run'.format(self.raw_data.N,self.raw_data.T*1e3,self.raw_data.Ec*1e6,self.raw_data.Gt*1e6,'varying',
                                                                                                                                                            self.raw_data.Ninterval)+r'$\times$'+'datapoint)={}, runs/datapoint={}, transient interval={}'.format(self.raw_data.Nruns,self.raw_data.number_of_concurrent,self.raw_data.Ntransient))
        else:
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
        if iterable(self.q0):
            plt.title('Results for N={}, T={:.1e} mK, Ec={:.1e} $\mu$eV, \n Gt={:.1e} $\mu$Si, q0={}'.format(self.raw_data.N,self.raw_data.T*1e3,self.raw_data.Ec*1e6,self.raw_data.Gt*1e6,'varying'))
        else:
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
    def savedata(self,filename=None,full_results=False):
        """

        stores all the input parameters and outputs of the simulation in a folder
        which by default is the same as the folder in which the plots are stored.

        Parameters
        ----------
        filename : str, optional
            folder/filename of the folder to store the data. The default is None.

        Returns
        -------
        None.

        """
        if full_results:
            if filename is None:
                try:
                    folder=os.getcwd()+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)
                    print('saving data in folder: '+str(folder))
                    np.savez_compressed(folder+'all_input_and_output_{}.npz'.format(self.raw_data.now),dQ=self.dQ,
                                        dt=self.dt,T=self.raw_data.T,Gt=self.raw_data.Gt,Ec=self.raw_data.Ec,
                                        N=self.raw_data.N,Nruns=self.raw_data.Nruns,Ninterval=self.raw_data.Ninterval,
                                        Ntransient=self.raw_data.Ntransient,q0=self.raw_data.q0,simulation_time=self.raw_data.simulation_time,
                                        now=self.raw_data.now,parallelization=self.raw_data.parallelization,batchsize=self.raw_data.batchsize,
                                        number_of_concurrent=self.raw_data.number_of_concurrent,V=self.raw_data.U,Gsm=self.Gsm,Gstd=self.Gstd,
                                        currents=self.currents,currentsm=self.currentsm,currentsstd=self.currentsstd,Gs=self.Gs,offset_C=self.raw_data.offset_C,
                                        second_order_C=self.raw_data.second_order_C,dC=self.raw_data.dC)
                except FileNotFoundError:
                    filepath=os.getcwd()
                    os.mkdir(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time))
                    np.savez_compressed(folder+'all_input_and_output_{}.npz'.format(self.raw_data.now),dQ=self.dQ,
                                        dt=self.dt,T=self.raw_data.T,Gt=self.raw_data.Gt,Ec=self.raw_data.Ec,
                                        N=self.raw_data.N,Nruns=self.raw_data.Nruns,Ninterval=self.raw_data.Ninterval,
                                        Ntransient=self.raw_data.Ntransient,q0=self.raw_data.q0,simulation_time=self.raw_data.simulation_time,
                                        now=self.raw_data.now,parallelization=self.raw_data.parallelization,batchsize=self.raw_data.batchsize,
                                        number_of_concurrent=self.raw_data.number_of_concurrent,V=self.raw_data.U,Gsm=self.Gsm,Gstd=self.Gstd,
                                        currents=self.currents,currentsm=self.currentsm,currentsstd=self.currentsstd,Gs=self.Gs,offset_C=self.raw_data.offset_C,
                                        second_order_C=self.raw_data.second_order_C,dC=self.raw_data.dC)
            else:
                print('saving data in: '+str(filename))
                np.savez_compressed(filename,dQ=self.dQ,
                                    dt=self.dt,T=self.raw_data.T,Gt=self.raw_data.Gt,Ec=self.raw_data.Ec,
                                    N=self.raw_data.N,Nruns=self.raw_data.Nruns,Ninterval=self.raw_data.Ninterval,
                                    Ntransient=self.raw_data.Ntransient,q0=self.raw_data.q0,simulation_time=self.raw_data.simulation_time,
                                    now=self.raw_data.now,parallelization=self.raw_data.parallelization,batchsize=self.raw_data.batchsize,
                                    number_of_concurrent=self.raw_data.number_of_concurrent,V=self.raw_data.U,Gsm=self.Gsm,Gstd=self.Gstd
                                    ,currents=self.currents,currentsm=self.currentsm,currentsstd=self.currentsstd,Gs=self.Gs,offset_C=self.raw_data.offset_C,
                                    second_order_C=self.raw_data.second_order_C,dC=self.raw_data.dC)
        else:
            if filename is None:
                try:
                    folder=os.getcwd()+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time)
                    print('saving data in folder: '+str(folder))
                    np.savez_compressed(folder+'all_input_and_output_{}.npz'.format(self.raw_data.now),
                                        T=self.raw_data.T,Gt=self.raw_data.Gt,Ec=self.raw_data.Ec,
                                        N=self.raw_data.N,Nruns=self.raw_data.Nruns,Ninterval=self.raw_data.Ninterval,
                                        Ntransient=self.raw_data.Ntransient,q0=self.raw_data.q0,simulation_time=self.raw_data.simulation_time,
                                        now=self.raw_data.now,parallelization=self.raw_data.parallelization,batchsize=self.raw_data.batchsize,
                                        number_of_concurrent=self.raw_data.number_of_concurrent,V=self.raw_data.U,Gsm=self.Gsm,Gstd=self.Gstd,
                                        currentsm=self.currentsm,currentsstd=self.currentsstd,offset_C=self.raw_data.offset_C,
                                        second_order_C=self.raw_data.second_order_C,dC=self.raw_data.dC)
                except FileNotFoundError:
                    filepath=os.getcwd()
                    os.mkdir(filepath+'\\Results {}, sim time={:.1f}sec\\'.format(self.now,self.simulation_time))
                    np.savez_compressed(folder+'all_input_and_output_{}.npz'.format(self.raw_data.now),
                                        T=self.raw_data.T,Gt=self.raw_data.Gt,Ec=self.raw_data.Ec,
                                        N=self.raw_data.N,Nruns=self.raw_data.Nruns,Ninterval=self.raw_data.Ninterval,
                                        Ntransient=self.raw_data.Ntransient,q0=self.raw_data.q0,simulation_time=self.raw_data.simulation_time,
                                        now=self.raw_data.now,parallelization=self.raw_data.parallelization,batchsize=self.raw_data.batchsize,
                                        number_of_concurrent=self.raw_data.number_of_concurrent,V=self.raw_data.U,Gsm=self.Gsm,Gstd=self.Gstd,
                                        currentsm=self.currentsm,currentsstd=self.currentsstd,offset_C=self.raw_data.offset_C,
                                        second_order_C=self.raw_data.second_order_C,dC=self.raw_data.dC)
            else:
                print('saving data in: '+str(filename))
                np.savez_compressed(filename,T=self.raw_data.T,Gt=self.raw_data.Gt,Ec=self.raw_data.Ec,
                                    N=self.raw_data.N,Nruns=self.raw_data.Nruns,Ninterval=self.raw_data.Ninterval,
                                    Ntransient=self.raw_data.Ntransient,q0=self.raw_data.q0,simulation_time=self.raw_data.simulation_time,
                                    now=self.raw_data.now,parallelization=self.raw_data.parallelization,batchsize=self.raw_data.batchsize,
                                    number_of_concurrent=self.raw_data.number_of_concurrent,V=self.raw_data.U,Gsm=self.Gsm,Gstd=self.Gstd,
                                    currents=self.currents,currentsm=self.currentsm,currentsstd=self.currentsstd,Gs=self.Gs,offset_C=self.raw_data.offset_C)

def carlo_CBT(U,T,Ec,Gt,N=100,Nruns=5000,Ntransient=5000,number_of_concurrent=5,Ninterval=1000,skip_transient=True,parallelization='external',
             n0=None,second_order_C=None,dtype='float64',offset_C=None,dC=0,n_jobs=2,batchsize=1,q0=0,split_voltage=True,dV=None,
             make_plots=False,save_plots=True,output='full',transient=10):
    """


    Parameters
    ----------
    U : 1d array of floats
        voltages at which to calculate teh conductance (not the current) using the monte carlo simulation.
        The simulation will actually be run for voltages shifted slightly relative to each value in order
        to calculate the conductance as the derivative of the current.

    T : float
        temperature in Kelvin.

    Ec : float
        charging energy in eV.

    Gt : float
        tunneling conductance in units of 1/ohm. Note that this just scales the final result,
        so there is no need to run the simulation multible times if one wants to check the influence of this parameter.

    N : int, optional
        number of islands in the CBT chain. The default is 100.

    Nruns : int, optional
        number of monte carlo steps to be taken for each charge array. The default is 5000.

    Ntransient : int, optional
        number of transient steps where no data is stored. This is just used to initialize the initial condition for each voltage. The default is 5000.

    number_of_concurrent : int, optional
        This corresponds to how many datapoints will end up being generated for each voltage.
        For a given voltage, each datapoint is generated by evolving from the same initial charge condition
        which is found by evolving a single charge configuration through the transient regime. The default is 5.

    Ninterval : int, optional
        Number monte carlo steps between storing data for charge transfer and time.
        Nothing bad happens if it is too low since in the end only the
        average is used to calculate the current, whose variance is calculated
        from parallel runs, so the autocorrelation doesnt matter.
        The only thing that can happen is that a lot of redundant correlated
        data is generated. The default is 1000.

    skip_transient : bool, optional
        Whether or not to skip the data saving of the initial transient steps. This should just be true unless one is
        interested in the odd behaviour of the transient regime or something. The default is True.
    parallelization : str, optional
        "external": simulations for each voltage are run in parallel using the joblib library in batches of size given by the input batchsize.
        "internal": simulations for each voltage are run in parallel by fancy numpy vectorization, but uses a for-loop to iiterate over batches.
        "non": for loop structure is used to iterate over voltages.

        The only reason not to use "external" is if the jobliib library somehow fails or interferes with other code.

        The default is 'external'.
    n0 : 1D-array of float of size N-1, optional
        initial charge configuration. The default is zeros.
    second_order_C : array of float, optional
        coupling between next to nearest neighbours. The default is zeros.
    dtype : str, optional
        datatype of the arrays used in calculations; using float32 should be faster, but it raises a lot of floatingpointerrors,
        so it shouldnt be touched I think. The default is 'float64'.
    offset_C : 1D array of float, optional
        capacitances representing coupling of dots to external charges. The units are e/Ec. The default is just zeros.
    dC : float, or 1D array, optional
        variations in capacitances in units of e/Ec. The default is 0.
    n_jobs : int, optional
        number of processes to run in parallel by the joblib module. The default is 2.
    batchsize : int, optional
        number of processes to run in parallel using 'internal' numpy vectorization. The default is 1.
    q0 : int, optional
        charge offset in units of e. The default is 0.

    split_voltage : bool, optional
        If True the voltages, are split so that two closely spaced data points replaces one of the input points.
        This is done to calculate the local differential more precisely. The default is True.
    dV : float, optional
        differential voltage used in the calculation of the conductance from the current data.
        A low value increases the uncertainty of the conductance datapoints a lot,
        but a high value induces a systematic uncertainty. The default is 50 times smaller than the FWHM of the first order model.

    make_plots : bool, optional
        If true, the conductance and current data is plotted in the end. The default is False.
    save_plots : bool, optional
        Whether or not to save the plots. The default is True.
    output : str, optional
        The final output. The default is 'full', which yelds a results object containing everything.
        If not everything is needed, one may choose to just output the mean conductances,
        or smething else from the list: ['full','G_mean, G_std','I_mean, I_std','G','I']
    transient : int, optional
        The number of additional steps to skip in the arrays of charge transfers and transition times in the current calculation apart from the ones already skipped during the simulation
    Raises
    ------
    Exception
        If the specified output parameter is not understood, an exception is raised before running the simulation,
        so that the mistake can be corrected before starting the simulation in vain.

    Returns
    -------
    result object
        if output is "full": a results object is generated whose main attrubutes
        of interest are Gsm yielding the mean conductances (one value for each input voltage) and
        Gstd yielding the standard deviations of the conductances (not the standard deviations of the means).

    """

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
        Vs=U
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
    if split_voltage:
        result=CBT_data_analysis(raw_result, transient=transient)

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
    else:
        print('data analysis is not done since, the conductances cannot be calculated due to the option split_voltage is False. The raw_result will be used as output.')
        return raw_result
def chi(a,b,delta):
    return np.sum((a-b)**2/delta**2)

def fit_carlo(V_data,G_data,u=None,V_data_std=None,G_data_std=None,V=None,N=100,Nruns=30000,Ninterval=10,Ntransient=300000,n_jobs=2,number_of_concurrent=20,parallelization='external',
              q0=0,dV=None,batchsize=10,transient=500,offset_C=None,dC=0,second_order_C=None,plot=True,filename=None,p0=None,save=True,save_fig_folder=None):
        """


    Parameters
    ----------
    V_data : TYPE
        DESCRIPTION.
    G_data : TYPE
        DESCRIPTION.
    u : TYPE, optional
        DESCRIPTION. The default is None.
    V_data_std : TYPE, optional
        DESCRIPTION. The default is None.
    G_data_std : TYPE, optional
        DESCRIPTION. The default is None.
    V : TYPE, optional
        DESCRIPTION. The default is None.
    N : TYPE, optional
        DESCRIPTION. The default is 100.
    Nruns : TYPE, optional
        DESCRIPTION. The default is 30000.
    Ninterval : TYPE, optional
        DESCRIPTION. The default is 10.
    Ntransient : TYPE, optional
        DESCRIPTION. The default is 300000.
    n_jobs : TYPE, optional
        DESCRIPTION. The default is 2.
    number_of_concurrent : TYPE, optional
        DESCRIPTION. The default is 20.
    parallelization : TYPE, optional
        DESCRIPTION. The default is 'external'.
    q0 : TYPE, optional
        DESCRIPTION. The default is 0.
    dV : TYPE, optional
        DESCRIPTION. The default is None.
    batchsize : TYPE, optional
        DESCRIPTION. The default is 10.
    transient : TYPE, optional
        DESCRIPTION. The default is 500.
    offset_C : TYPE, optional
        DESCRIPTION. The default is None.
    dC : TYPE, optional
        DESCRIPTION. The default is 0.
    second_order_C : TYPE, optional
        DESCRIPTION. The default is None.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    filename : TYPE, optional
        DESCRIPTION. The default is None.
    p0 : TYPE, optional
        DESCRIPTION. The default is None.
    save : TYPE, optional
        DESCRIPTION. The default is True.
    save_fig_folder : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """


        if V_data_std is None:
            V_data_std=0*V_data
        if G_data_std is None:
            G_data_std=0*G_data
        if filename is None:
            if u is None:
                raise Exception("when no MC data file is provided for fitting, the simulation will run, and input u has to be specified")
            if V is None:
                points =131
                lim=5.2*5.439*N
                V=np.linspace(-lim,lim,points)
            if dV is None:
                dV=5.439*N/(u*50)
            print('running MC simulation for u='+str(u))
            res=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=n_jobs,number_of_concurrent=number_of_concurrent,
                          parallelization=parallelization,q0=q0,dV=dV,batchsize=batchsize,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
            res.savedata()
            ####store main results###
            mean_conductances=res.Gsm #mean conductance
            std_conductance=res.Gstd #standard deviation of conductance
            offset_C=res.raw_data.offset_C
            second_order_C=res.raw_data.second_order_C
            Ec0=res.raw_data.Ec
            Gt0=res.raw_data.Gt

        else:
            print('loading data from file')
            data=np.load(filename)
            mean_conductances=data['Gsm']
            std_conductance=data['Gstd'] #standard deviation of conductance
            Ec0=data['Ec']
            Gt0=data['Gt']
            uu=data['Ec']/(kB*data['T'])
            offset_C=data['offset_C']
            print('loaded offset_C='+str(offset_C))
            N=data['N']
            print('loaded N='+str(N))
            try:
                second_order_C=data['second_order_C']
                print('loaded second_order_C='+str(second_order_C))
            except KeyError:
                second_order_C=np.zeros((N,))
            Nruns=data['Nruns']
            print('loaded Nruns='+str(Nruns))
            Ninterval=data['Ninterval']
            print('loaded Ninterval='+str(Ninterval))
            Ntransient=data['Ntransient']
            print('loaded Ntransient='+str(Ntransient))
            number_of_concurrent=data['number_of_concurrent']
            print('loaded number_of_concurrent='+str(number_of_concurrent))
            q0=data['q0']
            try:
                dC=data['dC']
            except KeyError:
                print('dC was not saved, and cannot be loaded')
            if V is not None:
                print('Using reloaded value for voltages instead of the provided values')
            Us=data['V']
            points=int(len(Us)/2)
            V=(Us[points:2*points]+Us[0:points])/2
            if uu != u:
                print('WARNING: the data in '+str(filename)+' are not for the provided value of u='+str(u)+'. The value u='+str(uu)+' will used instead.')
                u=uu

        print(u)
        model=interp1d(V/Ec0,mean_conductances/Gt0,kind='linear',bounds_error=False,fill_value=(np.mean(mean_conductances[0:5]),np.mean(mean_conductances[-5::])))
        wacky_sigma=interp1d(V,std_conductance,kind='linear',bounds_error=False,fill_value=(np.mean(std_conductance[0:5]),np.mean(std_conductance[-5::])))
        def CBT_model_g(x):
            """


            Parameters
            ----------
            x : floats
                unitsless variable.

            Returns
            -------
            function used in CBT model


            """
            return (x*np.sinh(x)-4*np.sinh(x/2)**2)/(8*np.sinh(x/2)**4)
        def f0(V,Ec,Gt,V0,T):

            return Gt*(1-(Ec/(kB*T))*CBT_model_g((V_data-V0)/(N*kB*T)))
        if p0 is None:
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
                # if all(G_data_std>0):
                #     chi_model=chi(G_data,G_MC,G_data_std)
                wacky_weight=wacky_sigma(V_data/par[0])*par[1]/np.sqrt(number_of_concurrent)
                chi_model=chi(G_data,G_MC,np.sqrt(G_data_std**2+wacky_weight**2))
                # wacky_weight=wacky_sigma(V_data/par[0])*par[1]/np.sqrt(number_of_concurrent)
                # chi_0=chi(G_data,f0(V_data,*par0),np.mean(wacky_sigma(V_data/par[0])*par[1]/np.sqrt(number_of_concurrent)))
                if plot:
                    fig=plt.figure(figsize=(11,6))
                    plt.errorbar(V_data,G_data,fmt='.',label='experimental data',yerr=G_data_std,xerr=V_data_std,markersize=1.5)
                    plt.title('Best MC Fit parameters for u={:.2f}, <$q_0^2$>={:.2f}e: '.format(u,np.mean(q0**2))+' T={:.1f} mK'.format(1e3*par[0]/(u*kB))+'\n $G_T={:.1e}$'.format(par[1])+r' $\Omega^{-1}$'+' $E_c$={:.1e} $\mu$eV, $C_0$/C={:.2f}, <$\delta C^2$>/C={:.2f}'.format(1e6*par[0],np.mean(offset_C),np.mean(dC**2))
                              +r', <$C^{(2)}$>'+'/C={:.3f}'.format(np.mean(second_order_C)))
                    # plt.errorbar(V_data,G_MC,yerr=wacky_sigma(V_data/par[0])*par[2]/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')
                    plt.errorbar(V*par[0]/Ec0+par[2],mean_conductances*par[1],yerr=par[1]*std_conductance/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2/n={:.1f}$'.format(chi_model/len(V_data)),fmt='.',capsize=1)
                    plt.ylabel('conductance [Si]')
                    plt.xlabel('Bias voltage [V]')
                    plt.tight_layout()

                    plt.legend(loc=3)
                    if save:
                        if filename is None:
                            try:
                                fig.savefig(res.filepath+'Chi_sq_plot1.png')
                            except FileNotFoundError:
                                os.mkdir(res.filepath)
                                fig.savefig(res.filepath+'Chi_sq_plot1.png')

                            res.savedata()
                        else:
                            if save_fig_folder is None:
                                fig.savefig(os.getcwd()+'Chi_sq_plot1.png')
                            else:
                                fig.savefig(save_fig_folder+'Chi_sq_plot1.png')

                    fig=plt.figure(figsize=(11,6))
                    plt.errorbar(V_data,G_data,fmt='.',label='experimental data',yerr=G_data_std,xerr=V_data_std,markersize=1.5)
                    plt.title('Best MC Fit parameters for u={:.2f}, <$q_0^2$>={:.2f}e: '.format(u,np.mean(q0**2))+' T={:.1f} mK'.format(1e3*par[0]/(u*kB))+'\n $G_T={:.1e}$'.format(par[1])+r' $\Omega^{-1}$'+' $E_c$={:.1e} $\mu$eV'.format(1e6*par[0]))
                    # plt.errorbar(V_data,G_MC,yerr=wacky_sigma(V_data/par[0])*par[2]/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2={:.1f}$'.format(chi_model),fmt='.')
                    plt.errorbar(V*par[0]/Ec0+par[2],mean_conductances*par[1],yerr=par[1]*std_conductance/np.sqrt(number_of_concurrent),label='MC Simulation, best fit for u={:.2f}: '.format(u)+' $\chi^2/n={:.1f}$'.format(chi_model/len(V_data)),fmt='.',capsize=1.5,mew=0.5,mec='black',ecolor='black')

                    #plt.plot(V_data,res.CBT_model_G((V_data-par[2])/par[0])*par[1],label='1st order result for same parameters as the MC')
                    plt.plot(V_data,f0(V_data,*par0),label='1st order result for optimal first order parameters: T={:.1f}mK'.format(1e3*par0[3]))
                    plt.legend(loc=3)
                    plt.ylabel('conductance [Si]')
                    plt.xlabel('Bias voltage [V]')
                    plt.tight_layout()
                    if filename is None:
                        try:
                            if save:
                                res.plotG(save=True)
                            else:
                                res.plotG()
                        except Exception:
                            pass #not important
                    if save:
                        if filename is None:
                            try:
                                fig.savefig(res.filepath+'Chi_sq_plot.png')
                            except FileNotFoundError:
                                os.mkdir(res.filepath)
                                fig.savefig(res.filepath+'Chi_sq_plot.png')

                            res.savedata()
                        else:
                            if save_fig_folder is None:
                                fig.savefig(os.getcwd()+'Chi_sq_plot.png')
                            else:
                                fig.savefig(save_fig_folder+'Chi_sq_plot.png')


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
        class fit_results:
            def __init__(self,fig=None):
                self.fit_model=f
                self.par=par
                self.wacky_sigma=wacky_sigma
                self.chi_model=chi_model
                self.wacky_weight=wacky_weight
                self.fig=fig
                self.offset_C=offset_C
                self.u=u
                self.second_order_C=second_order_C
                self.N=N
                self.dC=dC
                self.model=model
                self.q0=q0
        if plot:
            fit_result=fit_results(fig)
        else:
            fit_result=fit_results()
        return fit_result


#%%
if __name__=='__main__': #runs only if the file is being run explicitly
    pass
    ###################################################For testing########################################

    N=100 #Number of islands
    Ec=4e-6 #Charging energy in units of eV
    Gt=2e-5 #Large voltage asymptotic conductance (affects noly the scaling of the result)
    T=0.01 #Temperature in Kelvin
    FWHM=5.439*kB*T*N #Full width half max according to the first order model

    points=101 #number of voltages to run the simulation for
    lim=3*FWHM
    V=np.linspace(-lim,lim,points)


    ####Run main simulation####
    print('runing example')
    #res=carlo_CBT(V,T,Ec,Gt,N=100,Nruns=2000,Ninterval=10,Ntransient=10000,n_jobs=2,parallelization='internal',number_of_concurrent=15)
    print('finished running example')

    # ####store main results###
    # print('making plots')
    # mean_conductances=res.Gsm #mean conductance
    # std_conductance=res.Gstd #standard deviation of conductance
    # mean_currents=res.currentsm #mean currents
    # res.plotG(save=True)
    # res.plotI(save=True)
#%%
if __name__=='__main__':
    ####New example###
    lim=5*5.439*N
    V=np.linspace(-lim,lim,points)
    number_of_concurrent=12
    q0=0*np.random.uniform(low=-1,high=1,size=(N-1,))
    Nruns=30000
    Ninterval=10
    Ntransient=300000
    transient=500
    offset_C=0*np.ones((N-1,))/4
    dC=0#np.random.uniform(low=-5e-1,high=5e-1,size=(N,))
    second_order_C=np.ones((N,))/4
    u=2.5
    res=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)

    second_order_C=0*np.ones((N,))/4
    offset_C=0*np.ones((N-1,))/4
    res2=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)

    offset_C=np.ones((N-1,))/4
    res3=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
    offset_C=0*np.ones((N-1,))/4
    q0=np.random.uniform(low=-1,high=1,size=(N-1,))
    res4=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)

    q0=0*np.random.uniform(low=-1,high=1,size=(N-1,))
    dC=np.random.uniform(low=-5e-1,high=5e-1,size=(N,))
    res5=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
    q0=1/2 #0*np.random.uniform(low=-1,high=1,size=(N-1,))
    dC=0*np.random.uniform(low=-5e-1,high=5e-1,size=(N,))
    res6=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)

    np.seterr(all = 'warn')
    plt.figure(figsize=(11,6))
    V=np.linspace(-5,5,points)
    plt.errorbar(V,res2.Gsm,yerr=res2.Gstd,label='without any effects',linewidth=3)
    plt.errorbar(V,res.Gsm,yerr=res.Gstd,fmt='--',label='with second order effects')
    plt.errorbar(V,res3.Gsm,yerr=res3.Gstd,label='with offset C effects')
    plt.errorbar(V,res4.Gsm,yerr=res4.Gstd,label='with random offset charge effects')
    plt.errorbar(V,res5.Gsm,yerr=res5.Gstd,label='with random R and C effects')
    plt.errorbar(V,res6.Gsm,yerr=res5.Gstd,label='with q0=+1/2')
    Vs=np.linspace(-lim,lim,12000)
    plt.plot(Vs/(5.439*N),res.CBT_model_G(Vs),label='first order approximation',linewidth=3)

    plt.plot([0],[1-u/6+u**2/60-u**3/630],'o',label='third order prediction for the minimum')
    plt.xlabel('Voltage [$FWHM_0$]')
    plt.ylabel('G/$G_T$')
    plt.legend(loc=3)
    plt.grid()
    plt.tight_layout()
#%%
if __name__=='__main__':
    ####New example###
    lim=5*5.439*N
    V=np.linspace(-lim,lim,points)
    number_of_concurrent=15
    q0=0*np.random.uniform(low=-1,high=1,size=(N-1,))
    Nruns=30000
    Ninterval=5
    Ntransient=300000
    transient=1000
    offset_C=0*np.ones((N-1,))/4
    dC=0#np.random.uniform(low=-5e-1,high=5e-1,size=(N,))
    second_order_C=0*np.ones((N,))/4
    dC=np.random.uniform(low=-5e-1,high=5e-1,size=(N,))
    u=3
    res=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
    res.plotG()
    dC=0
    res2=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
    res2.plotG()
    dC=np.random.uniform(low=-5e-1,high=5e-1,size=(N,))
    u=4
    res3=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
    res3.plotG()
    dC=0
    res4=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
    res4.plotG()
    dC=np.random.uniform(low=-5e-1,high=5e-1,size=(N,))
    u=5
    res5=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)
    res5.plotG()
    dC=0
    res6=carlo_CBT(V,1/kB,u,1,N=N,Nruns=Nruns,Ninterval=Ninterval,Ntransient=Ntransient,n_jobs=2,number_of_concurrent=number_of_concurrent,
                  parallelization='external',q0=q0,dV=5.439*N/(u*50),batchsize=10,transient=transient,offset_C=offset_C,dC=dC,second_order_C=second_order_C)

    res6.plotG()
    np.seterr(all = 'warn')
    plt.figure(figsize=(11,6))
    V=np.linspace(-5,5,points)
    plt.errorbar(V,res2.Gsm,yerr=res2.Gstd,label='u=3, dC finite',linewidth=3)
    plt.errorbar(V,res.Gsm,yerr=res.Gstd,fmt='--',label='u=3 dC zero')
    plt.errorbar(V,res3.Gsm,yerr=res3.Gstd,label='u=4, dC finite')
    plt.errorbar(V,res4.Gsm,yerr=res4.Gstd,label='u=4, dC zero')
    plt.errorbar(V,res5.Gsm,yerr=res5.Gstd,label='u=5, dC finite')
    plt.errorbar(V,res6.Gsm,yerr=res5.Gstd,label='u=5, dC zero')
    Vs=np.linspace(-lim,lim,12000)
    # plt.plot(Vs/(5.439*N),res5.CBT_model_G(Vs),label='first order approximation u={:.1f}'.format(res5.raw_data.u),linewidth=3)

    # plt.plot([0],[1-u/6+u**2/60-u**3/630],'o',label='third order prediction for the minimum')
    plt.xlabel('Voltage [$FWHM_0$]')
    plt.ylabel('G/$G_T$')
    plt.legend(loc=3)
    plt.grid()
    plt.tight_layout()
