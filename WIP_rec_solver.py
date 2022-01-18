# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:22:39 2022

@author: Elias Roos Hansen
"""
import numpy as np
import matplotlib.pyplot as plt
def dE(n,U):
    return 1/2+n+U/2

def gam(n,b,U):
    n=n-1e-11
    U=U-1e-11
    b=b-1e-11
    return -dE(n,U)/np.expm1(-b*dE(n,U))
def CC(n,b,U):
    return gam(n, b, -U) + gam(n, b, U)+gam(-n, b, -U) + gam(-n, b, U)
def BB(n,b,U):
    return gam(n + 1, b, -U) + gam(n + 1, b, U)
def LL(n,b,U):
    return gam(-(n - 1), b, -U) + gam(-(n - 1), b, U)
# class recc:
#     def __init__(self):
#         self.A=0
#         self.n0=0
#     def run_rec(self,i,b,U):

#         def rec(n,b,U):
    
#             C=CC(n-1,b,U)
#             B=BB(n-1,b,U)
#             L=LL(n-1,b,U)
#             if n>0:
#                 if n>=self.n0+2:
#                     print(n)
                    
#                     # ns.append(n)
#                     self.A=-L*rec(n-2,b,U)/B+C*rec(n-1,b,U)/B
#                     return self.A
#                 elif n>=self.n0+1:
#                     print('nn '+str(n))
#                     self.A=C/((B+L))
#                     return C/((B+L))
#                 elif n>=self.n0:
#                     print('nnn '+str(n))
#                     self.A=1
#                     return 1
#             else:
#                 if n>=self.n0+2:
#                     print(n)
                    
#                     # ns.append(n)
#                     self.A=-L*rec(n-2,b,U)/B+C*rec(n-1,b,U)/B
#                     return self.A
#                 elif n>=self.n0+1:
#                     print('nn '+str(n))
#                     self.A=CC(1,b,U)/(BB(1,b,U)+LL(1,b,U))
#                     return self.A
#                 elif n>=self.n0:
#                     print('nnn '+str(n))
#                     self.A=1
#                     return 1
#         rec(i,b,U)
#         return self.A

# recc=recc()
# s=[recc.run_rec(n,0.1,0) for n in np.arange(15)]
# plt.plot(s)

class take2:
    def __init__(self,b,U):
        self.b=b
        self.U=U
        self.sigs=[1]

        sig1=CC(1,b,U)/(BB(1,b,U)+LL(1,b,U))
        self.sigs.append(sig1)
        self.sigs.insert(0,sig1)
        self.n=1
        self.ns=[-1,0,1]
        
    def coef1(self,n):
        U=self.U
        b=self.b
        C=CC(n-1,b,U)
        B=BB(n-1,b,U)
        L=LL(n-1,b,U)
        return C,B,L
    def coef2(self,n):
        U=self.U
        b=self.b
        C=CC(n+1,b,U)
        B=BB(n+1,b,U)
        L=LL(n+1,b,U)
        return C,B,L
    def nextsig(self):
        self.n+=1
        C,B,L=self.coef1(self.n)
        sign=-L*self.sigs[-2]/B+C*self.sigs[-1]/B
        self.sigs.append(sign)
        self.ns.append(self.n)
    def previoussig(self):
        self.n-=1
        C,B,L=self.coef2(self.n)
        sign=-B*self.sigs[1]/L+C*self.sigs[0]/L
        self.sigs.insert(0,sign)
        self.ns.insert(0,self.n)
    def __call__(self,number_of_ns,both_directions=True):
        
        if both_directions:
            sigtot=[]
            ntot=[]
            for _ in np.arange(np.abs(number_of_ns)):
                self.nextsig()
            sigtot=sigtot+self.sigs
            ntot=ntot+self.ns
            self.__init__(self.b,self.U)
            self.n+=-2
            for _ in np.arange(np.abs(number_of_ns)):
                self.previoussig()
            ntot=self.ns[0:-2]+ntot
            sigtot=self.sigs[0:-2]+sigtot
            return sigtot,ntot
        else:
            if number_of_ns>=0:
                for _ in np.arange(number_of_ns):
                    self.nextsig()
            else:
                self.n+=-2
                for _ in np.arange(np.abs(number_of_ns)):
                    self.previoussig()
            return self.sigs,self.ns
# recc=take2(1,-15)
# s,ns=recc(15)
# plt.plot(ns,s)
from scipy.integrate import solve_ivp
def diff(b,U,number_of_ns,t_eval=None):
    
    def D(n,sig):
        C=CC(n,b,U)
        B=BB(n,b,U)
        L=LL(n,b,U)
        sig0prime=-2*(B-L)*sig[0]/(B+L)-2*(B+L-C)*sig[1]/(B+L)
        sig1prime=sig[0]
        return np.array([sig0prime,sig1prime])
    y0=[0,1]
    res1=solve_ivp(D,(0,number_of_ns),y0=y0,t_eval=t_eval)
    res2=solve_ivp(D,(0,-number_of_ns),y0=y0,t_eval=-t_eval)

    y=np.empty((2,len(res1.y[0])+len(res2.y[0])))
    t=np.empty((1,len(res1.y[0])+len(res2.y[0])))
    y[:,0:len(res1.y[0])]=res1.y
    y[:,len(res1.y[0])::]=res2.y
    t[:,0:len(res1.y[0])]=res1.t
    t[:,len(res1.y[0])::]=res2.t
    ind=np.argsort(t)

    yy=y[:,ind[0]]

    tt=t[:,ind[0]]
    return yy,tt.flatten()
def diff_4(b,U,number_of_ns,t_eval=None):
    
    def D(n,sig):
        C=CC(n,b,U)
        B=BB(n,b,U)
        L=LL(n,b,U)
        k1=(B-L)/(B+L)
        k2=1-C/(B+L)
        sig0prime=-12*sig[1]-24*k1*(sig[2]+sig[0]/6)-24*k2*sig[3]
        sig1prime=sig[0]
        sig2prime=sig[1]
        sig3prime=sig[2]
        return np.array([sig0prime,sig1prime,sig2prime,sig3prime])
    C=CC(0,b,U)
    B=BB(0,b,U)
    L=LL(0,b,U)
    k2=1-C/(B+L)
    y0=[0,-k2,0,1]
    res1=solve_ivp(D,(0,number_of_ns),y0=y0,t_eval=t_eval)
    res2=solve_ivp(D,(0,-number_of_ns),y0=y0,t_eval=-t_eval)

    y=np.empty((4,len(res1.y[0])+len(res2.y[0])))
    t=np.empty((1,len(res1.y[0])+len(res2.y[0])))
    y[:,0:len(res1.y[0])]=res1.y
    y[:,len(res1.y[0])::]=res2.y
    t[:,0:len(res1.y[0])]=res1.t
    t[:,len(res1.y[0])::]=res2.t
    ind=np.argsort(t)

    yy=y[:,ind[0]]

    tt=t[:,ind[0]]
    return yy,tt.flatten()
def diff_0(b,U,number_of_ns,t_eval=None):
    def D(n,sig):
        sig0prime=-b*n*sig[0]-b*sig[1]
        sig1prime=sig[0]
        return np.array([sig0prime,sig1prime])
    y0=[0,1]
    res1=solve_ivp(D,(0,number_of_ns),y0=y0,t_eval=t_eval)
    res2=solve_ivp(D,(0,-number_of_ns),y0=y0,t_eval=-t_eval)


    y=np.empty((2,len(res1.y[0])+len(res2.y[0])))
    t=np.empty((1,len(res1.y[0])+len(res2.y[0])))
    y[:,0:len(res1.y[0])]=res1.y
    y[:,len(res1.y[0])::]=res2.y
    t[:,0:len(res1.y[0])]=res1.t
    t[:,len(res1.y[0])::]=res2.t
    ind=np.argsort(t)

    yy=y[:,ind[0]]

    tt=t[:,ind[0]]
    return yy,tt.flatten()
N=6
b=10
U=1
plt.figure(figsize=(11,9))
y,t=diff_0(b,U,N,t_eval=np.linspace(0,N-0.1,100))
plt.plot(t,y[1],label='Solution to differential equation expanded to first order in u (gaussian)')
y,t=diff(b,U,N,t_eval=np.linspace(0,N-0.1,100))
plt.plot(t,y[1],label='Solution to differential equation to all orders in u')
y,t=diff_4(b,U,N,t_eval=np.linspace(0,N-0.1,100))
plt.plot(t,y[-1],label='Solution to 4th differential equation to all orders in u')
# plt.plot(res2.t,res2.y[1])
recc=take2(b,U)
s,ns=recc(N)
plt.plot(ns,s,'.',label='Exact solution to master recurrence equation')
plt.ylabel('P (unnormalized)')
plt.xlabel('n')
plt.legend()
plt.tight_layout()
