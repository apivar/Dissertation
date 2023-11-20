#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is part of the code created for the dissertation of André Piva Romeu
Title: SHAPE AND TOPOLOGY OPTIMIZATION OF COMPLIANT MECHANISMS
ACTIVATED BY PIEZOCERAMICS
avaliable at: https://github.com/apivar/Dissertation
"""
import numpy as np
from ufl import tanh
from MMA import mmasub
from fenics import *
from dolfin_adjoint import *
from input_data import *
from Equations import *

pdat=problem_data()
fdat=filter_data()
adat=material_data()
mdat=mesh_data()

#Rectangular mesh with 4 boundaries
class UpperBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], mdat.Y_t)
class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], mdat.X_t)
#Extraboundaries, change if needed
class Output(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], mdat.Y_t) and between(x[0],\
                                     (mdat.X_t*.9,mdat.X_t))
class ExtraBoundary1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and between(x[0],(mdat.X_t*0.8,mdat.Y_t))
class ExtraBoundary2(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], mdat.Y_t) and between(x[0],(mdat.X_t*0.45,mdat.X_t*0.55))

#this part marks the output region for integrals later on, if there's it needs to be here    
output=Output()
contour=MeshFunction('size_t',mdat.mesh, mdat.mesh.topology().dim()-1)
contour.set_all(0)
output.mark(contour,1)
dsCont=ds(subdomain_data=contour)
#initial values and functions for the pseudo densities
chi_0= interpolate(pdat.chi_0,mdat.M)
rho_0= interpolate(pdat.rho_0,mdat.M)
rho  = Function(mdat.M,name="PseudoDensity")
chi  = Function(mdat.M,name="Material")
#both Boundary Conditions, W.sub(0) is displacement and W.sub(1) is Electric Potential
Zero =Constant(0.)
bc1  = [DirichletBC(mdat.W.sub(0).sub(0),Zero,LeftBoundary()),\
        DirichletBC(mdat.W.sub(0).sub(1),Zero,ExtraBoundary1()),\
        DirichletBC(mdat.W.sub(0).sub(0),Zero,ExtraBoundary1()),\
        DirichletBC(mdat.W.sub(1),pdat.volt, UpperBoundary()),\
        DirichletBC(mdat.W.sub(1),Zero,LowerBoundary())]
bc2  = [DirichletBC(mdat.W.sub(0).sub(0),Zero,LeftBoundary()),\
        DirichletBC(mdat.W.sub(0).sub(1),Zero,ExtraBoundary1()),\
        DirichletBC(mdat.W.sub(0).sub(0),Zero,ExtraBoundary1()),\
        DirichletBC(mdat.W.sub(1),Zero, UpperBoundary()),\
        DirichletBC(mdat.W.sub(1),Zero, LowerBoundary())]
#Output files
Folder   = "Results{0}x{1}".format(mdat.nx,mdat.ny)                                
rho_file = File(Folder+"/rho.pvd")
u_file   = File(Folder+"/u.pvd")
chi_file = File(Folder+"/chi.pvd")

def Forward(x1,x2,switch,beta):
    rho.vector()[:] = x2
    chi.vector()[:] = x1
    w = Function(mdat.W)
    w2 = Function(mdat.W)
    (U, V) = TrialFunctions(mdat.W)
    (utest, vtest) = TestFunctions(mdat.W)
    chi_tilde,rho_tilde=FilterUpdate(chi,rho,mdat.M,fdat,beta)
    C_pq,varepsilon_ik,e_kp,e_iq=Pemap(chi_tilde,rho_tilde,pdat,adat)
    F=Wf(U,V,utest,vtest,C_pq,varepsilon_ik,e_kp,e_iq)
    #First problem - original
    bilinear1 = dot(Constant((0.0,0.0)),utest)*dx
    problem1=LinearVariationalProblem(F,bilinear1,w,bcs=bc1)
    solver1=LinearVariationalSolver(problem1)
    solver1.parameters['linear_solver'] = 'umfpack'
    solver1.solve()
    u1 = Function(mdat.M,name="Displacement")
    phi1 = Function(mdat.M,name="Phi")
    (u1, phi1) = Function.split(w)
    u_file<<u1
    #Second problem - Compliance
    bilinear2 = dot(pdat.DummyLoadNegative, utest)*dsCont(1)
    problem2=LinearVariationalProblem(F,bilinear2,w2,bcs=bc2)
    solver2=LinearVariationalSolver(problem2)
    solver2.parameters['linear_solver'] = 'umfpack'
    solver2.solve()
    (u3, phi2) = Function.split(w2)
    #Section 4.3 - Objective Function
    #Mean transduction - equation 4.6
    L_2=assemble(dot(pdat.DummyLoad,u1)*dsCont(1))
    #Mean Compliance - equation 4.7
    L_3=assemble(dot(pdat.DummyLoadNegative,u3)*dsCont(1))
    #Minimize function - equation 4.9
    f_min=L_3/L_2   
    f0val=10e-4*f_min
    if switch==1:
        control = Control(chi)
        #Volume restriction - equation 4.12
        vc = UFLInequalityConstraint((chi-pdat.FracVol)*dx, control)
        fval = np.array(vc.function(chi), dtype=float)[np.newaxis][np.newaxis]
    elif switch==2:
        control = Control(rho)
        #Volume restriction - equation 4.13
        vc = UFLInequalityConstraint((rho-pdat.FracVol_Pzt)*dx, control)
        fval = np.array(vc.function(rho), dtype=float)[np.newaxis][np.newaxis]
    else:
        print("error in Forward value of Switch") 
    #Adjoint method computation
    df0dx_func = compute_gradient(f0val, control)
    df0dx = df0dx_func.vector().get_local()[np.newaxis].T
    dfdx = np.array(vc.jacobian(control.vector().get_local()))
    set_working_tape(Tape())
    
    return f0val, df0dx, fval, dfdx,u1

if __name__ == "__main__":
    x_2 = rho_0.vector().get_local()
    x_1 = chi_0.vector().get_local()
    beta=fdat.beta_0
    n = x_1.size
    m = 1                               
    xmin = np.zeros((n,1))
    xmax = np.ones((n,1)) 
    xval_1 = x_1[np.newaxis].T                  
    xold1_1 = xval_1.copy()                     
    xold2_1 = xval_1.copy()                     
    low_1 = np.ones((n,1))
    upp_1 = np.ones((n,1))
    #values eplained at section 3.6 - Solvers
    a0 = 1
    a = np.zeros((m,1)) 
    c = np.ones((m,1))*1e12
    d = np.ones((m,1))
    move = 0.3
    change = 1
    loop = -1
    Obj_1 = np.zeros((pdat.iterations,1))
    Obj_2 = np.zeros((pdat.iterations,1))
    L_2v = np.zeros((pdat.iterations,1))
    L_3v = np.zeros((pdat.iterations,1))
    chan = np.zeros((pdat.iterations,1))
    fpzt = np.zeros((pdat.iterations,1))
    fiso = np.zeros((pdat.iterations,1))
    Objchange = 0
    xval_2 = x_2[np.newaxis].T                  
    xold1_2 = xval_2.copy()                     
    xold2_2 = xval_2.copy()                     
    low_2 = np.ones((n,1))
    upp_2 = np.ones((n,1))
    while (change > pdat.change_min ) and ((loop+1)<pdat.iterations) and (Objchange<5):
        loop = loop+1
        if loop%fdat.beta_rot==0:
            if beta*2<fdat.beta_max:
                beta=beta*2
            else:
                beta=pdat.beta_max
        f0val_1,df0dx_1,fval_1,dfdx_1,ufinal=Forward(x_1,x_2,1,beta)
        xval_1 = x_1.copy()[np.newaxis].T
        xmma_1,ymma,zmma,lam,xsi,eta,mu_mma,zet,s,low_1,upp_1 = \
            mmasub(m,n,loop,xval_1,xmin,xmax,xold1_1,xold2_1,f0val_1,\
            df0dx_1,fval_1,dfdx_1,low_1,upp_1,a0,a,c,d,move)      
        xold2_1 = xold1_1.copy()
        xold1_1 = xval_1.copy()
        x_1 = xmma_1.copy().flatten()
        change_1 = np.linalg.norm(x_1.reshape(n,1)-xold1_1.reshape(n,1),np.inf)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("1: it.:{0} , obj.:{1:.3f}, ch.:{2:.3f}"\
              .format(loop,f0val_1,change_1))
        Obj_1[loop] = f0val_1
        if loop>1:
            ChObj=abs(Obj_1[loop-1]-Obj_1[loop-2])/abs(Obj_1[loop-1])
            if ChObj<pdat.ChObj_min:
                Objchange=Objchange+1
            else:
                Objchange=0
        f0val_2,df0dx_2,fval_2,dfdx_2,ufinal=Forward(x_1,x_2,2,beta)
        xval_2 = x_2.copy()[np.newaxis].T
        xmma_2,ymma,zmma,lam,xsi,eta,mu_mma,zet,s,low_2,upp_2 = \
            mmasub(m,n,loop,xval_2,xmin,xmax,xold1_2,xold2_2,f0val_2,\
            df0dx_2,fval_2,dfdx_2,low_2,upp_2,a0,a,c,d,move)      
        xold2_2 = xold1_2.copy()
        xold1_2 = xval_2.copy()
        x_2 = xmma_2.copy().flatten()
        change_2 = np.linalg.norm(x_2.reshape(n,1)-xold1_2.reshape(n,1),np.inf)
        change = max(change_1,change_2)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("2: it.:{0} , obj.:{1:.3f}, ch.:{2:.3f}"\
              .format(loop,f0val_2,change_2))
        rho.vector()[:] = x_2
        chi.vector()[:] = x_1
        chi_file << chi
        rho_file << rho
