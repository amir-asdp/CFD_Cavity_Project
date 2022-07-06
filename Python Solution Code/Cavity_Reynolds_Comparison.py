#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import torch as T
import time

from google.colab import files


# In[ ]:


class Cavity:
    def __init__(self, L = 1, H = 1, Re = 100, nx = 101, ny=101, U_up =1, U_down = 0, U_left = 0, U_right = 0):
        self.L = L
        self.H = H
        self.Re = Re
        self.nx = nx
        self.ny = ny

        self.U_up = U_up
        self.U_down = U_down
        self.U_left = U_left
        self.U_right = U_right

        self.nu = U_up * L / Re
        
        self.delta_x = L / (nx - 1)
        self.delta_y = H / (ny - 1)
        self.beta =self.delta_x / self.delta_y
        self.beta2 = self.beta ** 2

        self.sf = np.zeros((nx,ny))
        self.w = np.zeros((nx,ny))
        self.u = np.zeros((nx,ny))
        self.v = np.zeros((nx,ny))

        self.u[0,:] = U_down 
        self.u[-1,:] = U_up 
        self.v[:,0] = U_left
        self.v[:,-1] = U_right

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.sf_residuals = []
        self.w_residuals = []
        self.u_residuals = []
        self.v_residuals = []

        self.sf_imgs = []
        self.w_imgs  = []
        self.v_imgs  = []
        self.u_imgs  = []
        self.V_total = []

        self.step = 0

    def sf_update_roll_T(self):
        sf = T.tensor(self.sf,device=self.device)
        w = T.tensor(self.w, device=self.device)
        sf[1:-1,1:-1] = ((T.roll(sf,1,0) + T.roll(sf,-1,0) +                        self.beta2 * T.roll(sf,1,1) + self.beta2* T.roll(sf,-1,1) +                        (self.delta_x ** 2) * w) / (2 + 2*self.beta2))[1:-1,1:-1]

        return sf.cpu().detach().numpy()

    def w_update_roll_T(self):
        sf = T.tensor(self.sf, device=self.device)
        w = T.tensor(self.w, device=self.device)
        u = T.tensor(self.u, device=self.device)
        v = T.tensor(self.v, device=self.device)

        w[0,:] = -2 * (sf[1,:]+ self.delta_x * u[0,:]) / (self.delta_x **2)
        w[-1,:] = -2 * (sf[-2,:]+ self.delta_x * u[-1,:]) / (self.delta_x **2)
        w[:,0] = -2 * (sf[:,1] + self.delta_y * v[:,0]) / (self.delta_y **2)
        w[:,-1] = -2 * (sf[:,-2] + self.delta_y * v[:,-1]) / (self.delta_y **2)


        u[1:-1,1:-1]  = ((T.roll(sf,1,1) - T.roll(sf,-1,1)) / (2 * self.delta_y))[1:-1,1:-1] 
        v[1:-1,1:-1] = -((T.roll(sf,1,0) - T.roll(sf,-1,0)) / (2 * self.delta_x))[1:-1,1:-1] 

        dwdx = (T.roll(w,1,0) - T.roll(w,-1,0))/(2 * self.delta_x)
        dwdy = (T.roll(w,1,1) - T.roll(w,-1,1))/(2 * self.delta_y)

        A = (u * dwdx + v * dwdy)/ self.nu

        w[1:-1,1:-1] = ((T.roll(w,1,0) + T.roll(w,-1,0) + self.beta2*T.roll(w,1,1) +                         self.beta2 * T.roll(w,-1,1)- (self.delta_x**2) * A)/(2 + 2*self.beta2))[1:-1,1:-1]

        return w.cpu().detach().numpy(), u.cpu().detach().numpy(), v.cpu().detach().numpy()

    def solve(self,residual_target, frame):
        self.residual_target = residual_target

        Error = 1000

        startTime = time.time()
        while (Error > self.residual_target):
            sf_new = self.sf_update_roll_T()

            w_new, u_new, v_new = self.w_update_roll_T()

            error_sf = np.max(np.abs(sf_new - self.sf))
            error_w  = np.max(np.abs(w_new  - self.w))
            error_u  = np.max(np.abs(u_new  - self.u))
            error_v  = np.max(np.abs(v_new  - self.v))

            Error = max([error_sf, error_w, error_v, error_u])
            print("{} - {:.4e}      {:.4e}      {:.4e}      {:.4e}".format(self.step, error_sf,error_w,error_u, error_v))

            self.sf = sf_new
            self.w = w_new
            self.u = u_new
            self.v = v_new

            self.sf_residuals.append(error_sf)
            self.w_residuals.append(error_w)
            self.v_residuals.append(error_v)
            self.u_residuals.append(error_u)

            if (self.step % frame == 0):
                self.sf_imgs.append(self.sf)
                self.w_imgs.append(self.w)
                self.u_imgs.append(self.u)
                self.v_imgs.append(self.v)
                self.V_total.append( np.sqrt(self.u**2 + self.v**2))

            self.step += 1

        print('---------------------------')
        print("Run time: {:e}".format(time.time() -startTime))

    def plot_residuals(self):
        plt.figure(figsize=(10,5))
        plt.yscale('log')
        plt.plot(self.sf_residuals,label='stream function residual')
        plt.plot(self.w_residuals, label='vorticity residual')
        plt.plot(self.u_residuals, label='u residual')
        plt.plot(self.v_residuals, label='v residual')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('Error')
        _ = plt.title("solution residuals")

    def plot_stream(self):
        plt.figure(figsize=(12,10))
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.contour(self.sf, 50,colors='k',linewidths=0.8)
        plt.contourf(self.sf,50,cmap='jet')
        plt.title("Stream function - Re:{}, {}x{}".format(self.Re,self.nx, self.ny))
        plt.colorbar()

    def plot_vorticity(self):
        plt.figure(figsize=(12,10))
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.contour(self.w,1000,colors='k',linewidths=0.8)
        plt.contourf(self.w,1000,cmap='hot')
        plt.title("Vorticity")
        plt.colorbar()

    def plot_u(self):
        plt.figure(figsize=(12,10))
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.contour(self.u,100,colors='k',linewidths=0.8)
        plt.contourf(self.u,100,cmap='jet')
        plt.title("u velocity")

    def plot_v(self):
        plt.figure(figsize=(12,10))
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        plt.contour(self.v,100,colors='k',linewidths=0.8)
        plt.contourf(self.v,100,cmap='jet')
        plt.title("v velocity")

    def plot_Velocity(self):
        plt.figure(figsize=(12,10))
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        V_total = np.sqrt(self.u**2 + self.v**2)
        plt.contour(V_total, 100, colors='k',linewidths=0.8)
        plt.contourf(V_total, 100, cmap='jet')
        plt.title("Velocity Magnitude")

    def plot_streamlines(self):
        plt.figure(figsize=(12,10))
        plt.rcParams['contour.negative_linestyle'] = 'solid'

        x = np.linspace(0, C.H, self.ny)
        y = np.linspace(0, C.L, self.nx)

        X, Y = np.meshgrid(x, y)

        self.V_total = np.sqrt(self.u**2 + self.v**2)
        plt.streamplot(X, Y, C.v, C.u , density=2 , color =self.V_total, cmap='hot')
        plt.title("Streamlines - Re:{}, {}x{}".format(self.Re,self.nx, self.ny))
        plt.colorbar()


# # Re = 100

# In[ ]:


Re = 100
nx = 101
ny = 101
C = Cavity(Re=Re,nx=nx,ny=ny)
C.solve(1e-05,10000)


# In[ ]:


C.plot_residuals()
C.plot_stream()
C.plot_streamlines()


# # Re =400

# In[ ]:


Re = 400
nx = 201
ny = 201
C = Cavity(Re=Re,nx=nx,ny=ny)
C.solve(1e-05,10000)


# In[ ]:


C.plot_residuals()
C.plot_stream()
C.plot_streamlines()


# # Re =1000

# In[ ]:


Re = 1000
nx = 251
ny = 251
C = Cavity(Re=Re,nx=nx,ny=ny)
C.solve(1e-05,10000)


# In[ ]:


C.plot_residuals()
C.plot_stream()
C.plot_streamlines()
plt.savefig('Re1000.svg')
files.download("Re1000.svg")


# # LxH comp

# In[ ]:


L = 2
H = 1
C = Cavity(L = L, H = H, Re = 100 ,nx =int(L *100) + 1, ny = int(H *100) + 1)
C.solve(1e-05,10000)
C.plot_streamlines()
plt.title('Re = 100, H={}, L={}'.format(L,H))
plt.savefig("{}x{}.svg".format(L,H))
files.download("{}x{}.svg".format(L,H))


# In[ ]:


L = 1.5
H = 1
C = Cavity(L = L, H = H, Re = 100 ,nx =int(L *100) + 1, ny = int(H *100) + 1)
C.solve(1e-05,10000)
C.plot_streamlines()
plt.title('Re = 100, H={}, L={}'.format(L,H))
plt.savefig("{}x{}.svg".format(L,H))
files.download("{}x{}.svg".format(L,H))


# In[ ]:


L = 1
H = 1
C = Cavity(L = L, H = H, Re = 100 ,nx =int(L *100) + 1, ny = int(H *100) + 1)
C.solve(1e-05,10000)
C.plot_streamlines()
plt.title('Re = 100, H={}, L={}'.format(L,H))
plt.savefig("{}x{}.svg".format(L,H))
files.download("{}x{}.svg".format(L,H))


# In[ ]:


L = 1
H = 1.5
C = Cavity(L = L, H = H, Re = 100 ,nx =int(L *100) + 1, ny = int(H *100) + 1)
C.solve(1e-05,10000)
C.plot_streamlines()
plt.title('Re = 100, H={}, L={}'.format(L,H))
plt.savefig("{}x{}.svg".format(L,H))
files.download("{}x{}.svg".format(L,H))


# In[ ]:


L = 1
H = 2
C = Cavity(L = L, H = H, Re = 100 ,nx =int(L *100) + 1, ny = int(H *100) + 1)
C.solve(1e-05,10000)
C.plot_streamlines()
plt.title('Re = 100, H={}, L={}'.format(L,H))
plt.savefig("{}x{}.svg".format(L,H))
files.download("{}x{}.svg".format(L,H))


# # plot line

# In[ ]:


L = 1
H = 1

C3 = Cavity(L = L, H = H, Re = 100 ,nx =201,ny=201)
C3.solve(5e-04,10000)


# In[ ]:


C3.solve(4e-04,10000)


# In[ ]:


C1.u.shape[0]/2


# In[ ]:


x = [0, 0.0547, 0.0625, 0.0703, 0.1016,0.1719,0.2813,0.4531,0.5,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766,1]

y = [0.0, -0.03717, -0.04192, -0.04775, -.06434, -0.10150, -0.15662, -0.2109, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1]

plt.figure(figsize=(15,10))
plt.plot(x,y,'r.-',label='experiment')
plt.plot(np.linspace(0,1,C3.nx)[:-2],C3.v[:,int(C3.v.shape[0]/2)][:-2],label='cfd')
plt.legend()
plt.title("Results for v-Velociy along Horizantal Line")
plt.xlabel("x")
plt.ylabel("v")

plt.savefig("v.svg")
files.download("v.svg")


# In[ ]:



x = [1, 0.9688 ,0.9609 ,0.9531 ,0.9453 ,0.9063 ,0.8594 ,0.8047 ,0.5000 ,0.2344 ,0.2266 ,0.1563 ,0.0938 ,0.0781 ,0.0703 ,0.0625  ,0.0000 ]
y = [0.00000 ,-0.05906 ,-0.07391 ,-0.08864 ,-0.10313 ,-0.16914 ,-0.22445 ,-0.24533 ,0.05454 ,0.17527 ,0.17507 ,0.16077 ,0.12317 ,0.10890 ,0.10091 ,0.09233 ,0.0]
plt.figure(figsize=(15,10))
plt.plot(x,y,'r.-',label='experiment')
plt.plot(np.linspace(0,1,C3.nx),C3.u[int(C3.u.shape[0]/2),:],label='cfd')
plt.legend()
plt.title("Results for u-Velociy along Horizantal Line")
plt.xlabel("x")
plt.ylabel("u")

plt.savefig("u.svg")
files.download("u.svg")


# In[ ]:




