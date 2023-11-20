#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is part of the code created for the dissertation of André Piva Romeu
Title: SHAPE AND TOPOLOGY OPTIMIZATION OF COMPLIANT MECHANISMS
ACTIVATED BY PIEZOCERAMICS
avaliable at: https://github.com/apivar/Dissertation
"""

import numpy as np
from fenics import *
from dolfin_adjoint import * 
from Equations import *
class mesh_data:
    def __init__(self):
        self.X_t=50
        self.Y_t=100
        self.nx=40
        self.ny=80
        Origin=Point(0.0,0.0)
        Corner=Point(self.X_t,self.Y_t)
        self.mesh=RectangleMesh(Origin,Corner,self.nx,self.ny,'left/right')
        Ue = VectorElement('CG', self.mesh.ufl_cell(),2)
        Ve = FiniteElement('CG', self.mesh.ufl_cell(),1)
        self.W=FunctionSpace(self.mesh, MixedElement([Ue, Ve]))
        self.M=FunctionSpace(self.mesh,'CG', 1)
        
class material_data:
    def __init__(self):
        theta_piezo=0
        c11=c22=12.6
        c12=c21=7.95
        c31=c13=c23=c32=8.41
        c33=11.7
        c44=c55=2.3
        c66=2.325
        C =np.array([[c11,c12,c13,0,0,0],
                     [c21,c22,c23,0,0,0],
                     [c31,c32,c33,0,0,0],
                     [0,0,0,c44,0,0],
                     [0,0,0,0,c55,0],
                     [0,0,0,0,0,c66]])
        C=C*1e4
        C_r=Rotate(-pi/2, 'x', C)
        e_31=e_32=-6.5
        e_33=23.3
        e_24=e_15=17
        e=np.array([[0,0,0,0,e_15,0],
                    [0,0,0,e_24,0,0],
                    [e_31,e_32,e_33,0,0,0]])
        e=e*1e-6
        e_r=Rotate(-pi/2, 'x', e)
        vareps_11=vareps_22= 1.452e-14
        vareps_33  = 1.496e-14
        vareps=np.array([[vareps_11,0,0],
                    [0,vareps_22,0],
                    [0,0,vareps_33]])
        vareps_r=Rotate(-pi/2, 'x', vareps)
        C_r,e_r,vareps_r= Reduce(C_r, e_r, vareps_r)
        self.c_PZT=as_tensor(Rotate(theta_piezo, 'y', C_r,'r'))
        self.e_PZT=as_tensor(Rotate(theta_piezo, 'y', e_r,'r'))
        self.vareps_PZT=as_tensor(Rotate(theta_piezo, 'y', vareps_r,'r'))
        self.c_min=self.c_PZT*1e-12
        self.e_min=self.e_PZT*1e-12
        self.vareps_min=self.vareps_PZT*1e-12
        theta_iso = 2*pi
        E  = 10.6e4
        nu = 0.3
        a = E/((1+nu)*(1-2*nu))
        Ce = np.array([[a*(1-nu),     a*nu,        0.0],
                      [    a*nu, a*(1-nu),        0.0],
                      [     0.0,      0.0, a*(0.5-nu)]])
        self.c_ISO=as_tensor(Rotate(theta_iso, 'y', Ce,'r'))
        

class filter_data:
    def __init__(self):
        self.r=0.2
        self.beta_0=1
        self.beta_max=150
        self.alpha=0.5
        self.beta_rot=40
        
class problem_data:
    def __init__(self):
        self.eta_0=3
        self.eta_1=3
        self.eta_2=1
        self.iterations=200
        self.volt=Constant(200.0)
        self.FracVol=0.5
        self.FracVol_Pzt=0.25
        self.chi_0=Constant(0.5)
        self.rho_0=Constant(0.25)
        self.change_min=0.001
        self.ChObj_min=0.001
        self.DummyLoad=Constant((0.0, 1.0))
        self.DummyLoadNegative=-self.DummyLoad
