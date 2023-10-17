#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:43:50 2023

@author: ancient
"""

import numpy as np
import matplotlib.pyplot as plt
from ufl import tanh, ln
from MMA import mmasub
from fenics import *
from dolfin_adjoint import *
from ufl import tanh

def e_j(u):
    return T2V_2D(epsilon(u))

def E_k(v):
    return Vector_2D(E_field(v))

def C_ij(csi,rho,pdat,adat):
    return csi**pdat.p0*(rho**pdat.p1*adat.c_pzt\
                  +(1-rho**pdat.p1)*adat.c_base)+1e-8*adat.c_base

def d_ij(csi,rho,pdat,d_pzt):
    return csi**pdat.p0*rho**pdat.p2*d_pzt

def d_ki(csi,rho,pdat,d_pzt_T):
    return csi**pdat.p0*rho**pdat.p2*d_pzt_T

def e_ik(csi,rho,pdat,e_pzt):
    return csi**pdat.p0*rho**pdat.p2*e_pzt

def sigma_i(u,v,csi,rho,pdat,adat):
    return V2T_2D(dot(C_ij(csi,rho,pdat,adat), e_j(u))\
                  -dot(d_ki(csi,rho,pdat,adat.d_pzt_T),E_k(v)))

def D_i(u,v,csi,rho,pdat,adat):
    return Vector_2D(dot(d_ij(csi,rho,pdat,adat.d_pzt),e_j(u))\
                  +dot(e_ik(csi,rho,pdat,adat.e_pzt),E_k(v)))

def Vector_2D(vec):
    return as_vector([vec[0], vec[1]])

def T2V_2D(ten):
    return as_vector([ten[0,0],ten[1,1],ten[0,1]])

def V2T_2D(vec):
    return as_tensor([[vec[0], vec[2]],
                      [vec[2], vec[1]]])
def E_field(phi):
    return -grad(phi)

def epsilon(u):
	return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def RotZ_2D(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    Tz = np.array([[  c**2,  s**2,       s*c],
                   [  s**2,  c**2,      -s*c],
                   [-2*s*c, 2*s*c, c**2-s**2]])
    Rz = np.array([[ c, s],
                   [-s, c]])
    return Tz, Rz

def Rotate_Z_3x3(a,theta):
    Tz, Rz = RotZ_2D(theta)
    return ((Tz.T).dot(a)).dot(Tz)

def Rotate_Z_2x2(a,theta):
    Tz, Rz = RotZ_2D(theta)
    return ((Rz.T).dot(a)).dot(Rz)

def Rotate_Z_2x3(a,theta):
    Tz, Rz = RotZ_2D(theta)
    return ((Rz.T).dot(a)).dot(Tz)

def helmholtz_filter(a,r,M):
    if r == 0:
        a_f = a
    else:
        a_f = TrialFunction(M)
        vH = TestFunction(M)
        F = (inner(r * r * grad(a_f), grad(vH)) * dx
             + inner(a_f, vH) * dx - inner(a, vH) * dx)
        aH = lhs(F)
        LH = rhs(F)
        a_f = Function(M, name="Filtered Control")
        solve(aH == LH, a_f)
    return a_f

def projecting(a,fdat,M):
    beta=fdat.beta
    eta=fdat.eta
    if beta == 0:
        a_f = a
    else:
        a_f = (tanh(beta*eta)+tanh(beta*(a-eta)))/(tanh(beta*eta)
                                        +tanh(beta*(1-eta)))
    return project(a_f, M)