#!/usr/bin/env python
"""
     _________    _
    \         \  \
    \         \  \ Y_t
    \_________\  \_
     _________
    \   X_t   \

"""
import numpy as np
from fenics import *
from dolfin_adjoint import * 
from Equations import *
class mesh_data:
    """Parameters to configure the FWI algorithm"""
    def __init__(self):
        # General Configuration ------------------------------------------------- #
        
        
        self.X_t=50
        self.Y_t=50
        nx=40
        ny=40
        self.delta_x=self.X_t/nx 
        self.delta_y=self.Y_t/ny
        Origin=Point(0.0,0.0)
        Corner=Point(self.X_t,self.Y_t)
        self.mesh=RectangleMesh(Origin,Corner,nx,ny,'left/right')
        Ue = VectorElement('CG', self.mesh.ufl_cell(),2) #(Vetor) Deslocamento 'u'
        Ve = FiniteElement('CG', self.mesh.ufl_cell(),1) #(Escalar) Voltagem 'phi'
        self.W=FunctionSpace(self.mesh, MixedElement([Ue, Ve]))
        self.M=FunctionSpace(self.mesh,'CG', 1)
        
class material_data:
    """Parameters to configure the FWI algorithm"""
    def __init__(self):
        # General Configuration ------------------------------------------------- #
        theta_piezo=pi
        c11  = 12.1e4
        c13  = 7.52e4
        c33  = 11.1e4
        c66  = 2.1e4
        d13  = -5.4e-6
        d33  = 15.8e-6
        d15  = 12.3e-6
        e11  = 1.452e-14
        e33  = 1.496e-14
        C = np.array([[c11, c13, 0.0],
                      [c13, c33, 0.0],
                      [0.0, 0.0, c66]])      
        d = np.array([[0.0, 0.0, d15],
                      [d13, d33, 0.0]])
        e = np.array([[e11, 0.0],
                      [0.0, e33]])
        self.c_pzt=as_tensor(Rotate_Z_3x3(C,theta_piezo))
        self.e_pzt=as_tensor(Rotate_Z_2x2(e,theta_piezo))
        self.d_pzt=as_tensor(Rotate_Z_2x3(d,theta_piezo))
        self.d_pzt_T=as_tensor((Rotate_Z_2x3(d,theta_piezo)).T)
        theta_aluminio = 2*pi
        E  = 10.6e4
        nu = 0.3
        a = E/((1+nu)*(1-2*nu))
        Ce = np.array([[a*(1-nu),     a*nu,        0.0],
                      [    a*nu, a*(1-nu),        0.0],
                      [     0.0,      0.0, a*(0.5-nu)]])
        self.c_base=as_tensor(Rotate_Z_3x3(Ce,theta_aluminio))

class filter_data:
    """Parameters to configure the FWI algorithm"""
    def __init__(self):
        # General Configuration ------------------------------------------------- #
        self.r=0
        self.beta=0
        self.max_beta=250
        self.eta=0.5
        
class problem_data:
    """Parameters to configure the FWI algorithm"""
    def __init__(self):
        self.p0=3
        self.p1=3
        self.p2=1
        self.iterations=10
        self.volt=Constant(20.0)
        self.FracVol=0.5
        self.FracVol_Pzt=0.25
        self.csi_0=Constant(0.5)
        self.rho_0=Constant(0.25)
        self.DummyLoad=Constant((-1.0, 0.0))
        self.DummyLoadNegative=-self.DummyLoad

