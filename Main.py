#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:05:11 2023

@author: ancient
"""
import numpy as np
import matplotlib.pyplot as plt
from ufl import tanh, ln
from MMA import mmasub
from fenics import *
from dolfin_adjoint import *
from ufl import tanh
from input_data import *
from Equations import *

pdat=problem_data()
fdat=filter_data()
adat=material_data()
mdat=mesh_data()

class BordaSuperior(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], mdat.Y_t)
class BordaInferior(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)
class BordaEsquerda(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and between(x[1],(0.0,(0.0+mdat.delta_y)))
class BordaDireita(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], mdat.X_t)
class Output(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], mdat.X_t) and between(x[1],\
                                     ((mdat.Y_t-mdat.delta_y),mdat.Y_t))
output=Output()
contour=MeshFunction('size_t',mdat.mesh, mdat.mesh.topology().dim()-1)
contour.set_all(0)
output.mark(contour,1)
dsCont=ds(subdomain_data=contour)

csi_0= interpolate(pdat.csi_0,mdat.M)
rho_0= interpolate(pdat.rho_0,mdat.M)
rho  = Function(mdat.M,name="PseudoDensity")
csi  = Function(mdat.M,name="Material")

Zero =Constant(0.)
bc1  = [DirichletBC(mdat.W.sub(0).sub(1),Zero,BordaSuperior()),\
        DirichletBC(mdat.W.sub(0).sub(1),Zero,BordaEsquerda()),\
        DirichletBC(mdat.W.sub(0).sub(0),Zero,BordaEsquerda()),\
        DirichletBC(mdat.W.sub(1),pdat.volt, BordaSuperior()),\
        DirichletBC(mdat.W.sub(1),Zero,BordaInferior())]
bc2  = [DirichletBC(mdat.W.sub(0).sub(1),Zero,BordaSuperior()),\
        DirichletBC(mdat.W.sub(0).sub(1),Zero,BordaEsquerda()),\
        DirichletBC(mdat.W.sub(0).sub(0),Zero,BordaEsquerda()),\
        DirichletBC(mdat.W.sub(1),Zero, BordaSuperior()),\
        DirichletBC(mdat.W.sub(1),Zero, BordaInferior())]

Folder   = "Results" 
rho_file = File(Folder+"/rho_iteracao.pvd")
u_file   = File(Folder+"/u_iteracao.pvd")
csi_file = File(Folder+"/csi_iteracao.pvd")
phi_file = File(Folder+"/phi_iteracao.pvd")
    
def Forward(x1,x2,switch):
    rho.vector()[:] = x2
    csi.vector()[:] = x1
    w = Function(mdat.W)
    w2 = Function(mdat.W)
    (U, V) = TrialFunctions(mdat.W)
    (utest, vtest) = TestFunctions(mdat.W)
    rho_hf = helmholtz_filter(rho,fdat.r,mdat.M)
    rho_pj = projecting(rho_hf,fdat,mdat.M)
    csi_hf = helmholtz_filter(csi, fdat.r,mdat.M)
    csi_pj = projecting(csi_hf,fdat,mdat.M)
    U_form = inner(sigma_i(U,V,csi_pj,rho_pj,pdat,adat), epsilon(utest))*dx
    V_form = inner(D_i(U,V,csi_pj,rho_pj,pdat,adat), grad(vtest))*dx #ver sinal!!
    F = U_form + V_form
    bilinear1 = dot(Constant((0.0,0.0)),utest)*dx
    bilinear2 = dot(pdat.DummyLoadNegative, utest)*dsCont(1)
    problem1=LinearVariationalProblem(F,bilinear1,w,bcs=bc1)
    solver1=LinearVariationalSolver(problem1)
    solver1.parameters['linear_solver'] = 'umfpack'
    solver1.solve()
    u1 = Function(mdat.M,name="Displacement")
    phi1 = Function(mdat.M,name="Phi")
    (u1, phi1) = Function.split(w)
    problem2=LinearVariationalProblem(F,bilinear2,w2,bcs=bc2)
    solver2=LinearVariationalSolver(problem2)
    solver2.parameters['linear_solver'] = 'umfpack'
    solver2.solve()
    (u3, phi2) = Function.split(w2)
    L2 = assemble(dot(pdat.DummyLoad,u1)*dsCont(1))
    L3 = assemble(dot(pdat.DummyLoadNegative,u3)*dsCont(1))
    rho_file << rho
    u_file << u1
    csi_file << csi
    phi_file << phi1
    f0val = -1e7*(L2/L3)
    #f0val =-1e7*L2
    if switch==1:
        control = Control(csi)
        vc = UFLInequalityConstraint((csi-pdat.FracVol)*dx, control)
        fval = np.array(vc.function(csi), dtype=float)[np.newaxis][np.newaxis]
    elif switch==2:
        control = Control(rho)
        vc = UFLInequalityConstraint((rho-pdat.FracVol_Pzt)*dx, control)
        fval = np.array(vc.function(rho), dtype=float)[np.newaxis][np.newaxis]
    else:
        print("erro no valor Forward(x1,x2, []")     
    df0dx_func = compute_gradient(f0val, control)
    df0dx = df0dx_func.vector().get_local()[np.newaxis].T
    dfdx = np.array(vc.jacobian(control.vector().get_local()))
    set_working_tape(Tape())
    frac_solid = assemble(csi*dx)/(mdat.X_t*mdat.Y_t)
    frac_pzt = assemble(rho*dx)/(mdat.X_t*mdat.Y_t)
    return f0val, df0dx, fval, dfdx, frac_solid, frac_pzt

if __name__ == "__main__":
    x_2 = rho_0.vector().get_local()
    x_1 = csi_0.vector().get_local()
    n = x_1.size
    m = 1                               
    xmin = np.zeros((n,1))
    xmax = np.ones((n,1)) 
    xval_1 = x_1[np.newaxis].T                  
    xold1_1 = xval_1.copy()                     
    xold2_1 = xval_1.copy()                     
    low_1 = np.ones((n,1))
    upp_1 = np.ones((n,1))
    a0 = 1 #pq?
    a = np.zeros((m,1)) 
    c = np.ones((m,1))*1e9 #ver se não deveria ser xzero
    d = np.ones((m,1)) #rever
    # d = np.zeros((m,1)) #rever
    move = 0.3
    change = 1
    loop = -1
    Obj_1 = np.zeros((pdat.iterations,1))
    xval_2 = x_2[np.newaxis].T                  
    xold1_2 = xval_2.copy()                     
    xold2_2 = xval_2.copy()                     
    low_2 = np.ones((n,1))
    upp_2 = np.ones((n,1))
    Obj_2 = np.zeros((pdat.iterations,1))
    while (change > 1e-5 ) and (loop<pdat.iterations):
        loop = loop+1
        f0val_1,df0dx_1,fval_1,dfdx_1,frac_solid,frac_pzt=Forward(x_1,x_2,1)
        xval_1 = x_1.copy()[np.newaxis].T
        xmma_1,ymma,zmma,lam,xsi,eta,mu_mma,zet,s,low_1,upp_1 = \
            mmasub(m,n,loop,xval_1,xmin,xmax,xold1_1,xold2_1,f0val_1,\
            df0dx_1,fval_1,dfdx_1,low_1,upp_1,a0,a,c,d,move)      
        xold2_1 = xold1_1.copy()
        xold1_1 = xval_1.copy()
        x_1 = xmma_1.copy().flatten()
        change_1 = np.linalg.norm(x_1.reshape(n,1)-xold1_1.reshape(n,1),np.inf)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("1: it.:{0} , obj.:{1:.3f}, ch.:{2:.3f}, f_solid:{3:.3f}, f_pzt:{4:.3f}"\
              .format(loop,f0val_1,change_1,frac_solid,frac_pzt))
        Obj_1[loop-1] = f0val_1
        
        f0val_2,df0dx_2,fval_2,dfdx_2,frac_solid,frac_pzt=Forward(x_1,x_2,2)
        xval_2 = x_2.copy()[np.newaxis].T
        xmma_2,ymma,zmma,lam,xsi,eta,mu_mma,zet,s,low_2,upp_2 = \
            mmasub(m,n,loop,xval_2,xmin,xmax,xold1_2,xold2_2,f0val_2,\
            df0dx_2,fval_2,dfdx_2,low_2,upp_2,a0,a,c,d,move)      
        xold2_2 = xold1_2.copy()
        xold1_2 = xval_2.copy()
        x_2 = xmma_2.copy().flatten()
        change_2 = np.linalg.norm(x_2.reshape(n,1)-xold1_2.reshape(n,1),np.inf)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("1: it.:{0} , obj.:{1:.3f}, ch.:{2:.3f}, f_solid:{3:.3f}, f_pzt:{4:.3f}"\
              .format(loop,f0val_2,change_2,frac_solid,frac_pzt))
        Obj_2[loop-1] = f0val_2
    end_1 = Function(mdat.M,name="PseudoDensity")
    end_1.vector()[:] = x_1
    plt.figure(1)
    plt.colorbar(plot(end_1, title='csi'))
    end_2 = Function(mdat.M,name="Piezoelectric")
    end_2.vector()[:] = x_2
    plt.figure(2)
    plt.colorbar(plot(end_2, title='rho'))
    plt.figure(3)
    plt.plot(Obj_1)
    plt.figure(4)
    plt.plot(Obj_2)
    with open(Folder+"/JporInt.txt", 'w') as f:
        f.write('Obj_1\n')
        f.write(str(Obj_2))