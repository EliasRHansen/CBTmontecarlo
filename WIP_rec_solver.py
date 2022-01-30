# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:22:39 2022

@author: Elias Roos Hansen
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import exprel,expm1
from derivative import dxdt
def dE(n,U):
    return 1/2+n+U/2

def gam(n,b,U):
    n=n-1e-16
    U=U-1e-16
    b=b-1e-16
    #return -dE(n,U)/np.expm1(-b*dE(n,U))
    
    return 1/(b*exprel(-b*dE(n,U)))
def CC(n,b,U):
    return gam(n, b, -U) + gam(n, b, U)+gam(-n, b, -U) + gam(-n, b, U)
def BB(n,b,U):
    return gam(n + 1, b, -U) + gam(n + 1, b, U)
def LL(n,b,U):
    return gam(-(n - 1), b, -U) + gam(-(n - 1), b, U)

#Schematically: BB(n)*P(n+1)+LL(n)*P(n-1)-C(n)*P(n)=0
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
        sig0=1
        self.sigs=[sig0]

        sig1=CC(0,b,U)*sig0/(BB(0,b,U)+LL(0,b,U))
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
        sig0prime=-2*(B-L)*sig[0]/(B+L)-2*(1-C/(B+L))*sig[1]
        sig1prime=sig[0]
        return np.array([sig0prime,sig1prime])
    y0=[0,1]
    res1=solve_ivp(D,(0,number_of_ns),y0=y0,t_eval=t_eval)
    res2=solve_ivp(D,(0,-number_of_ns),y0=y0,t_eval=-t_eval)

    y=np.empty((2,len(res1.y[0])+len(res2.y[0])-1))
    t=np.empty((1,len(res1.y[0])+len(res2.y[0])-1))
    y[:,0:len(res1.y[0])-1]=res1.y[:,1::]
    y[:,len(res1.y[0])-1::]=res2.y
    t[:,0:len(res1.y[0])-1]=res1.t[1::]
    t[:,len(res1.y[0])-1::]=res2.t
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
    y0=[0,-2*k2,0,1]
    res1=solve_ivp(D,(0,number_of_ns),y0=y0,t_eval=t_eval)
    res2=solve_ivp(D,(0,-number_of_ns),y0=y0,t_eval=-t_eval)

    y=np.empty((4,len(res1.y[0])+len(res2.y[0])-1))
    t=np.empty((1,len(res1.y[0])+len(res2.y[0])-1))
    y[:,0:len(res1.y[0])-1]=res1.y[:,1::]
    y[:,len(res1.y[0])-1::]=res2.y
    t[:,0:len(res1.y[0])-1]=res1.t[1::]
    t[:,len(res1.y[0])-1::]=res2.t
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


    y=np.empty((2,len(res1.y[0])+len(res2.y[0])-1))
    t=np.empty((1,len(res1.y[0])+len(res2.y[0])-1))
    y[:,0:len(res1.y[0])-1]=res1.y[:,1::]
    y[:,len(res2.y[0])-1::]=res2.y
    t[:,0:len(res1.y[0])-1]=res1.t[1::]
    t[:,len(res2.y[0])-1::]=res2.t
    ind=np.argsort(t)

    yy=y[:,ind[0]]

    tt=t[:,ind[0]]
    return yy,tt.flatten()
#%%
if __name__=='__main__':
    N=6 
    b=2
    U=0
    def analytic_1(b,U,ns):
        return np.exp(-b*ns**2/2)/np.sqrt(np.pi*2/b)
    def analytic_2(b,U,ns):
        return np.exp(-b*(1-5*b/12)*ns**2/2)*np.sqrt(b*(1-5*b/12)/(2*np.pi))
    def analytic_4(b,U,ns):
        # f=b*(1-5*b/12)+(1-ns**2+U**2/6)*b**3/8+(23*U**2/1440-17/576+31*ns**2/240-ns**4/11)*b**4
        f=b*(1-5*b/12)+(1+U**2/6)*b**3/8+(23*U**2/1440-17/576)*b**4
        return np.exp(-f*ns**2/2)*np.sqrt(f/(2*np.pi))
    plt.figure(figsize=(10,6))
    plt.ylim((-0.01,0.7))
    # y,t=diff_0(b,U,N,t_eval=np.linspace(0,N-0.1,100))
    # plt.plot(t,y[1]/simps(y[1],x=t),label='Solution to differential equation expanded to first order in u (gaussian)')
    # plt.plot(t,analytic_1(b,U,t),'--',label='Solution to 2nd order differential equation to first order in u (gaussian)')
    y,t=diff(b,U,N,t_eval=np.linspace(0,N-0.1,100))
    plt.plot(t,y[1]/simps(y[1],x=t),label='Solution to 2nd order differential equation to all orders in u')
    # plt.plot(t,analytic_1(b,U,t),'--',label='Solution to 2nd order differential equation to 1st order in u (gaussian)')
    # plt.plot(t,analytic_2(b,U,t),'--',label='Solution to 2nd order differential equation to 2nd order in u (gaussian)')
    #plt.plot(t,analytic_4(b,U,t),'--',label='Solution to 2nd order differential equation to 4th order in u (gaussian)')
    
    y,t=diff_4(b,U,N,t_eval=np.linspace(0,N-0.1,100))
    plt.plot(t,y[-1]/simps(np.abs(y[-1]),x=t),label='Solution to 4th order differential equation to all orders in u')
    # plt.plot(res2.t,res2.y[1])
    recc=take2(b,U)
    s,ns=recc(N)
    plt.plot(ns,s/np.sum(s),'o-',label='Exact solution to master recurrence equation',color='black')
    plt.ylabel('P (roughly normalized)')
    plt.xlabel('n')
    plt.title('Master eq. vs approximate 2nd order dif. eq. u={}, eU/Ec={}'.format(b,U))
    plt.legend(loc=1)
    plt.tight_layout()
#%%
if __name__=='__main__':
    N=7 
    b=2
    U=100
    def analytic_1(b,U,ns):
        return np.exp(-b*ns**2/2)/np.sqrt(np.pi*2/b)
    def analytic_2i(b,U,ns):
        v=1/b+5/12
        return np.exp(-ns**2/(2*v))*np.sqrt(1/(2*np.pi*v))
    def analytic_4i(b,U,ns):
        # f=b*(1-5*b/12)+(1-ns**2+U**2/6)*b**3/8+(23*U**2/1440-17/576+31*ns**2/240-ns**4/11)*b**4
        # v=1/b+5/12+((7+3*U**2)/144+ns**2/8)*b#+((-5+3*U**2)/2160-ns**2/40+ns**4/11)*b**2
        # v=1/b+5/12+((7+3*U**2)/144+ns**2/12)*b+((-5+3*U**2)/2160-ns**2/60)*b**2
        # v=1/b+5/12+((7+3*U**2)/144)*b+((-5+3*U**2)/2160)*b**2
        # v=1/b+5/12+((7+3*U**2)/144+ns**2/12)*b+((-5+3*U**2)/2160-ns**2/60)*b**2-((9*U**4+174*U**2+49)/103680+(18*U**2+34)*ns**2/8640+ns**4/40)*b**3
        
        # v=1/b+5/12+((7+3*U**2)/144+ns**2/12)*b+((-5+3*U**2)/2160-ns**2/60)*b**2-((9*U**4+174*U**2+49)/103680+(18*U**2+34)*ns**2/8640+ns**4/40)*b**3
        # v=v+((171*U**4+30*U**2+175)/2177280+(152*U**2+328)*ns**2/241920+269*ns**4/10080)*b**4
        
        v=1/b+5/12+((7+3*U**2)/144)*b+((-5+3*U**2)/2160)*b**2-((9*U**4+174*U**2+49)/103680)*b**3
        v=v+((171*U**4+30*U**2+175)/2177280)*b**4
        return np.exp(-ns**2/(2*v))
    plt.figure(figsize=(10,6))
    plt.ylim((-0.01,0.7))
    # y,t=diff_0(b,U,N,t_eval=np.linspace(0,N-0.1,100))
    # plt.plot(t,y[1]/simps(y[1],x=t),label='Solution to differential equation expanded to first order in u (gaussian)')
    # plt.plot(t,analytic_1(b,U,t),'--',label='Solution to 2nd order differential equation to first order in u (gaussian)')
    y,t=diff(b,U,N,t_eval=np.linspace(0,N-0.1,100))
    plt.plot(t,y[1]/simps(y[1],x=t),label='Solution to 2nd order differential equation to all orders in u')
    # plt.plot(t,analytic_1(b,U,t),'--',label='Solution to 2nd order differential equation to 1st order in u (gaussian)')
    # plt.plot(t,analytic_2(b,U,t),'--',label='Solution to 2nd order differential equation to 2nd order in u (gaussian)')
    plt.plot(t,analytic_2i(b,U,t),'--',label='Solution to 2nd order diff. equation to 2nd order in u (gaussian)')
    plt.plot(t,analytic_4i(b,U,t)/simps(analytic_4i(b,U,t),x=t),'--',label='Solution to 2nd order diff. eq. including 4th order terms (gaussian)')
    
    # y,t=diff_4(b,U,N,t_eval=np.linspace(0,N-0.1,100))
    # plt.plot(t,y[-1]/simps(np.abs(y[-1]),x=t),label='Solution to 4th order differential equation to all orders in u')
    # plt.plot(res2.t,res2.y[1])
    recc=take2(b,U)
    s,ns=recc(N)
    plt.plot(ns,s/np.sum(s),'o-',label='Exact solution to master recurrence equation',color='black')
    plt.ylabel('P (roughly normalized)')
    plt.xlabel('n')
    plt.title('Master eq. vs approximate 2nd order dif. eq. u={}, eU/Ec={}'.format(b,U))
    plt.legend(loc=1)
    plt.tight_layout()
#%%

#find full width half max as a function of 

# recc=take2(b,U)
# s,ns=recc(N)
def var0_4(b,U):
        v=1/b+5/12+((7+3*U**2)/144)*b+((-5+3*U**2)/2160)*b**2-((9*U**4+174*U**2+49)/103680)*b**3
        v=v+((171*U**4+30*U**2+175)/2177280)*b**4
        return v
def var0_2(b,U):
        v=1/b+5/12
        return v
kB=8.617*1e-5
bb=4e-6/(kB*np.linspace(1e-3,2000e-3,7000))

U=0
Us=np.linspace(0,25,251)
Vars=[]
Vars0_2=[]
Vars0_4=[]
N=40
for U in Us:
    print(U)
    var=[]
    vars0_2=[]
    vars0_4=[]
    # plt.figure(figsize=(10,6))
    # plt.title('Exact solution to master recurrence equation, eU/Ec={:.2f}'.format(U))
    for b_i in np.arange(len(bb)):
        b=bb[b_i]
        recc=take2(b,U)
        s,ns=recc(N)
        
        variance=np.sum(np.array(ns)**2*np.array(s))/np.sum(np.array(s))
        var.append(variance)
        vars0_2.append(var0_2(b,U))
        vars0_4.append(var0_4(b,U))
        # if b_i%1000==0:
        #     plt.plot(ns,s/np.sum(s),'o-',label='Exact solution to master recurrence equation, u={:.2f}'.format(b),color=[bb[-1]/b,0,0])
        #     plt.ylabel('P')
        #     plt.xlabel('charge [# of e]')
            # plt.pause(0.05)
            # plt.draw()
    Vars.append(var)
    Vars0_2.append(vars0_2)
    Vars0_4.append(vars0_4)
Vars=np.array(Vars)
Vars0_2=np.array(Vars0_2)
Vars0_4=np.array(Vars0_4)
# plt.figure(figsize=(10,6))
# for i in np.arange(len(Us)):
#     plt.plot(bb,bb*np.array(Vars[i]),label='exact, U/Ec={}'.format(Us[i]),color=[i/len(Us),0,0])   
#     plt.xlabel('u')
#     plt.ylabel(r'u$\times$ var')	
#     plt.title('variance vs u')
    
# plt.plot(bb,bb*np.array(Vars0_2[0]),'--',label='approximation',color=[0,i/len(Us),0]) 
# plt.legend()
plt.figure(figsize=(10,6))
dlnvardlnus=[]
for i in np.arange(len(Us)):
    dlnvardlnu=np.gradient(np.gradient(np.log(Vars[i]),np.log(bb),axis=-1,edge_order=2),np.log(bb),axis=-1,edge_order=2)
    dlnvardlnus.append(dlnvardlnu)
    plt.semilogx(bb,dlnvardlnu,'.-',label='exact, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)   
    plt.xlabel('u')
    plt.ylabel(r'$\frac{d^2ln(var)}{dln(u)^2}$')	
    plt.title('second derivative of variance')
dlnvardlnus=np.array(dlnvardlnus)
plt.semilogx(bb[0:-2],np.diff(np.diff(np.log(Vars0_2[0]))/np.diff(np.log(bb)))/np.diff(np.log(bb)[0:-1]),'--',label='approximation',color=[0,i/len(Us),0],linewidth=1) 
plt.legend()

plt.figure(figsize=(10,6))
for i in np.arange(len(Us)):
    plt.plot(bb,1/Vars[i],'.-',label='exact, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0])   
    plt.xlabel('u')
    # plt.ylabel(r'u$\times$ var')	
    plt.title('variance vs u')

plt.figure()
argmaxs=np.array([np.argmax(s) for s in dlnvardlnus])
maxu=bb[argmaxs]
plt.plot(Us,maxu)
plt.xlabel('eU/Ec')
plt.ylabel(r'u for which $\frac{d^2ln(var)}{dln(u)^2}$ is maximized')
plt.figure()
for b_i in np.arange(len(bb)):
    if b_i%1000==0:
        plt.figure()
        finalvar=Vars[:,b_i]
        
        plt.plot(Us,finalvar)
        plt.xlabel('eU/Ec')
        plt.ylabel(r'var')



from scipy.optimize import curve_fit
def varinv(u,m,m1,a,b):
    return -a*expm1(-m*u)-b*expm1(-m1*u**2)

plt.figure()
for i in np.arange(len(Us)):
    if i%10==0:
        p0=[1,1/(Vars[i][-1]),0]
        par,cov=curve_fit(varinv,bb,1/Vars[i],p0=p0)
        print(par)
        plt.plot(bb,1/Vars[i],'.-',label='exact, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)
        plt.plot(bb,varinv(bb,*par),label='fit, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)
def fitf(u,g,C,g1,D):
    return (u**g+D*u**g1)*np.exp(-u)*C
def fitf(u,g,C,m,E):
    return (u**g)*np.exp(-m*u)*C+E
# def fitf(u,b,c,d,e,m):
#     return (a+u)*np.exp(-m*u)
# def fitf(u,b,c,d,e,m):
#     return e*(b*u+c*u**2+d*u**3+u**4)*np.exp(-m*u)
def fitf(u,a,b,g,m):
    return b*(a+u**(2*g))*np.exp(-m*u**g)
def fitf(u,m,m1,a,b):
    return varinv(u,m,m1,a,b)
plt.figure()
for i in np.arange(len(Us)):
    # p0=[maxu[i],dlnvardlnus[i][argmaxs[i]],1/maxu[i],0]
    # p0=[maxu[i],dlnvardlnus[i][argmaxs[i]],1,0]
    if i%10==0:
        # p0=[0,0.5,-0.5,dlnvardlnus[i][argmaxs[i]],1]
        g0=3/2
        m0=3
        p0=[0,dlnvardlnus[i][argmaxs[i]]*(m0/g0)**g0*np.exp(g0),g0,m0]
        # p0=[0,dlnvardlnus[i][argmaxs[i]],0,0,1,0]
        par,cov=curve_fit(fitf,bb,dlnvardlnus[i],p0=p0)
        
        plt.plot(bb,dlnvardlnus[i],'.-',label='exact, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)
        plt.plot(bb,fitf(bb,*par),label='fit, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)

# def fitf2(U,A,B,g):
#     return A/(1+np.abs((U/B))**g)
# p0=[1,1,2]

# par,cov=curve_fit(fitf2,Us,maxu,p0=p0)
# plt.figure()
# plt.plot(Us,maxu)
# plt.plot(Us,fitf2(Us,*par))
# plt.xlabel('eU/Ec')
# plt.ylabel(r'u for which $\frac{d^2ln(var)}{dln(u)^2}$ is maximized')
# plt.figure()
# for i in np.arange(len(Us)):
#     plt.semilogx(bb,1/np.array(Vars0_4[i]),'--',label='approximation, U/Ec={}'.format(Us[i]),color=[0,i/len(Us),0])   
#     plt.xlabel('u')
#     plt.ylabel('1/var')	
plt.figure()
for i in np.arange(len(Us)):
    if i%10==0:
        lndvardu=np.gradient(np.log(np.gradient(1/Vars[i],bb,axis=-1,edge_order=2)),bb,axis=-1,edge_order=2)
        
        plt.semilogx(bb,lndvardu,'.-',label='exact, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)   
        plt.xlabel('u')
        plt.ylabel(r'$\frac{d}{du}ln\frac{d(1/var)}{d(u)}$')	
        plt.title('second derivative of variance')
plt.figure()
for i in np.arange(len(Us)):
    if i%1==0:
        # lndvardu=np.gradient(np.log(np.gradient(1/Vars[i],bb,axis=-1,edge_order=2)),bb,axis=-1,edge_order=2)
        
        plt.plot(bb,-(1-Vars[i][0]/Vars[i]),'.-',label='exact, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)   
        plt.xlabel('u')
        # plt.ylabel(r'$\frac{d}{du}ln\frac{d(1/var)}{d(u)}$')	
        # plt.title('second derivative of variance')
def varinv2(u,k1,u0,m):
    return k1*(u-u0)*np.exp(-m*u)/u0
plt.figure()
pars=[]
for i in np.arange(len(Us)):
    if i%1==0:
        p0=[1,1,1]
        par,cov=curve_fit(varinv2,bb,-(1-Vars[i][0]/Vars[i]),p0=p0)
        pars.append(par)
        print(par)
        plt.semilogx(bb,-(1-Vars[i][0]/Vars[i]),'.-',label='exact, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)
        plt.semilogx(bb,varinv2(bb,*par),label='fit, U/Ec={:.1f}'.format(Us[i]),color=[i/len(Us),0,0],linewidth=1)
plt.figure()
pars=np.array(pars)
plt.plot(Us,1/pars[:,1])
plt.figure()
plt.plot(Us[0:-1],np.diff(pars[:,2]))
plt.figure()

plt.plot(Us,Vars[:,0])
#%%

 
# plt.legend()
fig,ax=plt.subplots(figsize=(10,6))

Bs,UU=np.meshgrid(bb[0:-2],Us)
dlnvardlnu=np.diff(np.diff(np.log(Vars),axis=-1)/np.diff(np.log(bb)),axis=-1)/np.diff(np.log(bb)[0:-1])

plot=ax.pcolormesh(UU,Bs,dlnvardlnu)
ax.set_xlabel('Bias voltage [Ec/e]')
ax.set_ylabel('u')
cbar0=fig.colorbar(plot, ax=ax, extend='both')

