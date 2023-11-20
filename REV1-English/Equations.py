#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is part of the code created for the dissertation of André Piva Romeu
Title: SHAPE AND TOPOLOGY OPTIMIZATION OF COMPLIANT MECHANISMS
ACTIVATED BY PIEZOCERAMICS
avaliable at: https://github.com/apivar/Dissertation
"""
import numpy as np
from ufl import tanh, ln
from fenics import *
from dolfin_adjoint import *

#%% Section 3.2 - Piezoelectric Constitutive Equations
#Voigt notation - equation 3.8
#Transform from vector to tensor
def VectorToTensor_2D(vec):
    return as_tensor([[vec[0], vec[2]],
                      [vec[2], vec[1]]])
#Transform from tensor to vector
def TensorToVector_2D(ten):
    return as_vector([ten[0,0],ten[1,1],ten[0,1]])
#Transform from array to Fenics Variable
def Vector_2D(vec):
    return as_vector([vec[0], vec[1]])
#Mechanical stress tensor - equation 3.9a
def sigma_p(C_pq,S_q,e_kp,E_k):
    sigma_p=dot(C_pq, S_q)-dot(e_kp,E_k)
    return sigma_p
#Electric displacement vector - equation 3.9b
def D_i(S_q,e_iq,varepsilon_ik,E_k):
    D_i=dot(e_iq,S_q)+dot(varepsilon_ik,E_k)
    return D_i
#%% Section 3.2.1 - Rotation and Plane Stress
#Rotation matrices, equations 3.10 and 3.13
class RotMatrix:
    def __init__(self,theta,axis):
        self.axis=axis
        self.c=cos(theta)
        self.s=sin(theta)
        if axis=='x':
            self.K6=np.array([[1,             0,            0,                  0,     0,      0],
                              [0,     self.c**2,    self.s**2,    2*self.c*self.s,     0,      0],
                              [0,     self.s**2,    self.c**2,   -2*self.c*self.s,     0,      0],
                              [0,-self.c*self.s,self.c*self.s,self.c**2-self.s**2,     0,      0],
                              [0,             0,            0,                  0,self.c,-self.s],
                              [0,             0,            0,                  0,self.s, self.c]])
            self.K3=np.array([[1,0, 0],
                              [0,self.c,-self.s],
                              [0,self.s, self.c]])
        elif axis=='y':
            self.K6=np.array([[self.c**2,0,self.s**2,0,2*self.c*self.s,0],
                              [0,1,0,0, 0],
                              [self.s**2,0,self.c**2,0,-2*self.c*self.s,0],
                              [0,0,0,self.c,0,-self.s],
                              [-self.c*self.s,0,self.c*self.s,0,self.c**2-self.s**2,0],
                              [0,0,0,self.s,0,self.c]])
            self.K3=np.array([[self.c,0,self.s],
                              [0,1,0],
                              [-self.s,0,self.c]])
        elif axis=='z':
            self.K6=np.array([[self.c**2,self.s**2,0,0,0,2*self.c*self.s],
                              [self.s**2,self.c**2,0, 0,0,-2*self.c*self.s],
                              [0,0,1,0,0,0],
                              [0,0,0,self.c,self.s,0],
                              [0,0,0,-self.s,self.c,0],
                              [-self.c*self.s,self.c*self.s,0,0,0,self.c**2-self.s**2]])
            self.K3=np.array([[self.c,-self.s,0],
                              [self.s,self.c,0],
                              [0,0,1]])
        self.R=np.array([[self.c,self.s],
                    [-self.s,self.c]]) 
        self.T=np.array([[self.c**2,self.s**2,self.c*self.s],
                [self.s**2,self.c**2,-self.c*self.s],
                [-2*self.c*self.s,2*self.c*self.s,self.c**2-self.s**2]])
#How to rotate, equations present in table 3.1
#Input: angle or rotation (rad), axis or rotation (string, ie: 'x', 'y' or 'z'),
# tensor for rotation (np.array), compleat or reduced tensor ('c'or'r')       
def Rotate(theta,axis,matrix,state='c'): 
    Rt=RotMatrix(theta,axis)
    dimensions = np.shape(matrix)
    rows, columns = dimensions
    if state=='c':
        if rows==3:
            if columns==3:
                a=np.matmul(np.matmul(Rt.K3,matrix),Rt.K3.T)
            elif columns==6:
                a=np.matmul(np.matmul(Rt.K3,matrix),Rt.K6.T)
        if rows==6:
            if columns==3:
                a=np.matmul(np.matmul(Rt.K6,matrix),Rt.K3.T)
            elif columns==6:
                a=np.matmul(np.matmul(Rt.K6,matrix),Rt.K6.T)
    elif state=='r':
        if rows==3:
            if columns==3:
                a=np.matmul(np.matmul(Rt.T.T,matrix),Rt.T)
            elif columns==2:
                a=np.matmul(np.matmul(Rt.T.T,matrix),Rt.R)
        if rows==2:
            if columns==3:
                a=np.matmul(np.matmul(Rt.R.T,matrix),Rt.T)
            elif columns==2:
                a=np.matmul(np.matmul(Rt.R.T,matrix),Rt.R)
    return a
#Plane Stress reduction, equations 3.11
def Reduce(C,e,vareps):
    c_11=C[0,0]-(C[0,2]*C[0,2])/C[2,2]
    c_12=C[0,1]-(C[0,2]*C[1,2])/C[2,2]
    c_22=C[1,1]-(C[1,2]*C[1,2])/C[2,2]
    c_66=C[5,5]
    e_21=e[1,0]-(C[0,2]*e[2,2])/C[2,2]
    e_22=e[1,1]-(C[1,2]*e[2,2])/C[2,2]
    vareps_22=vareps[1,1]+(e[2,2]*e[2,2])/C[2,2]        
    C_r = np.array([[c_11, c_12, 0.0],
                    [c_12, c_22, 0.0],
                    [0.0, 0.0, c_66]])   
    e_r = np.array([[0.0, 0.0, e[0,5]],
                    [e_21, e_22, 0.0]])
    vareps_r = np.array([[vareps[0,0], 0.0],
                         [0.0,   vareps_22]])        
    return C_r,e_r,vareps_r 
#%% Section 3.4 - Topology Optimization
#Classical SIMP - equation 3.14
def Simp(a,rho):
       Y=a.Y_min+(a.Y_0-a.Y_min)*rho**a.eta
       return Y
#Dual SIMP - equation 3.15
def DualSimp(a,chi,rho):
    Y=chi**a.eta_a*(rho**a.eta_b*a.Y_b+(1-rho**a.eta_b)*a.Y_a)+(1-chi**a.eta_a)*a.Y_min
    return Y
#PEMAP - equations 3.16
def Pemap(chi,rho,pdat,adat):
    C_pq=chi**pdat.eta_0*(rho**pdat.eta_1*adat.c_PZT+(1-rho**pdat.eta_1)*adat.c_ISO)+(1-chi**pdat.eta_0)*adat.c_min
    varepsilon_ik=chi**pdat.eta_0*rho**pdat.eta_2*adat.vareps_PZT+(1-chi**pdat.eta_0)*adat.vareps_min
    e_iq=chi**pdat.eta_0*rho**pdat.eta_2*adat.e_PZT+(1-chi**pdat.eta_0)*adat.e_min
    e_kp=e_iq.T
    return C_pq,varepsilon_ik,e_kp,e_iq 
#%% Section 3.5 - Filters
#Helmholtz Type Filter - equations 3.17 and 3.18
#Input: pseudo density, strenght parameter, function space 
def HTF(pseudo,r,M):
    if r == 0:
        pseudo_hat = pseudo
    else:
        pseudo_hat = TrialFunction(M)
        vH = TestFunction(M)
        F = (inner(r*r*grad(pseudo_hat), grad(vH))*dx
             +inner(pseudo_hat,vH)*dx-inner(pseudo,vH)*dx)
        aH = lhs(F)
        LH = rhs(F)
        pseudo_hat = Function(M,name="Filtered Control")
        solve(aH == LH,pseudo_hat)
    return pseudo_hat
#Smooth-Heaviside Function - equation 3.19
def SHF(pseudo_hat,alpha,beta,M):
    if beta == 0:
        pseudo_tilde = pseudo_hat
    else:
        pseudo_tilde = (tanh(beta*alpha)+tanh(beta*(pseudo_hat-alpha)))/(tanh(beta*alpha)
                                        +tanh(beta*(1-alpha)))
    return project(pseudo_tilde, M)
#Mnd - equation 3.27
def Mnd(pseudo):
    Mnd= np.sum(4*pseudo*(1-pseudo)/np.size(pseudo))
    return Mnd
#Equation to update variables (HTF then SHF)
def FilterUpdate(chi,rho,M,fdat,beta):
    r=fdat.r
    alpha=fdat.alpha
    chi_filtered= SHF(HTF(chi,r,M),alpha,beta,M)
    rho_filtered= SHF(HTF(rho,r,M),alpha,beta,M)
    return chi_filtered,rho_filtered
#%% Section 4.1 - FEM Formulation
#Weak form - equation 4.3
def Wf(U,V,utest,vtest,C_pq,varepsilon_ik,e_kp,e_iq):
    Mec=inner(sigma(U,V,C_pq,e_kp),S(utest))*dx
    Elec=inner(D(U,V,varepsilon_ik,e_iq),E(vtest))*dx
    Wf=Mec+Elec
    return Wf
#Deformation tensor - equation 4.4a
def S(u):
    S=0.5*(nabla_grad(u) + nabla_grad(u).T)
    return S
def S_q(u):
    return TensorToVector_2D(S(u))
#Electric vector field - equation 4.4b
def E(phi):
    E=-grad(phi)
    return E
def E_k(v):
    return Vector_2D(E(v))
#Calculate sigma
def sigma(U,V,C_pq,e_kp):
    return VectorToTensor_2D(sigma_p(C_pq, S_q(U), e_kp, E_k(V)))
#Calculate D
def D(U,V,varepsilon_ik,e_kp):
    return Vector_2D(D_i(S_q(U),e_kp,varepsilon_ik,E_k(V)))

if __name__ == "__main__":
   print(teste)