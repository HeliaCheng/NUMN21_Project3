import numpy as np
import matplotlib.pyplot as plt


class Appartment(): #Maybe break up into different classes. Give better name?
    def __init__(self, delta_x = 1/20, initial_temp = 20, D = 0.1):
        self.delta_x = delta_x
        self.D = D
        self.omega = 0.8 #Used in relaxation  
        self.initial_temp = initial_temp
        self.n = int(1/delta_x)
        self.u1 = np.ones((self.n, self.n)) * self.initial_temp #the different rooms at t = 0 
        self.u2 = np.ones((2*self.n, self.n)) * self.initial_temp
        self.u3 = np.ones((self.n, self.n)) * self.initial_temp
        self.A = (np.eye(self.n,self.n)*(-4) + np.eye(self.n,self.n,1) + np.eye(self.n,self.n,-1) + np.eye(self.n,self.n,3) + np.eye(self.n,self.n,-3))/delta_x**2
        self.H = 40 #Heated
        self.WF = 5 #Window
        self.NW = 15#Normal
        #Walls room 1:
        self.north_wall_1 = np.ones(int(1/delta_x)) * self.NW
        self.east_wall_1 = np.ones(int(1/delta_x)) * self.NW  #Should depend on room 2
        self.south_wall_1 = np.ones(int(1/delta_x)) * self.NW
        self.west_wall_1 = np.ones(int(1/delta_x)) * self.H
        #Walls room 2:
        self.north_wall_2 = np.ones(int(1/delta_x)) * self.H
        self.east_wall_2 = np.ones(int(2/delta_x)) * self.NW #Should partially depend on room 3
        self.south_wall_2 = np.ones(int(1/delta_x)) * self.WF
        self.west_wall_2 = np.ones(int(2/delta_x)) * self.NW #Should partially depend on room 1
        #Walls room 3:
        self.north_wall_3 = np.ones(int(1/delta_x)) * self.NW
        self.east_wall_3 = np.ones(int(1/delta_x)) * self.H
        self.south_wall_3 = np.ones(int(1/delta_x)) * self.NW
        self.west_wall_3 = np.ones(int(1/delta_x)) * self.NW  #Should de pend on room 2

    def step(self):
        """
        Does one time step
        """
        self.east_wall_1 = self.u2[self.n:,0]
        self.west_wall_2[self.n:] = self.u1[:,-1]
        self.east_wall_2[:self.n] = self.u3[:,0]
        self.west_wall_3 = self.u2[:self.n,-1]
        un1 = self.heat_equation(self.u1, self.north_wall_1, self.south_wall_1, self.west_wall_1, self.east_wall_1, 1)
        un2 = self.heat_equation(self.u2, self.north_wall_2, self.south_wall_2, self.west_wall_2, self.east_wall_2, 2)
        un3 = self.heat_equation(self.u3, self.north_wall_3, self.south_wall_3, self.west_wall_3, self.east_wall_3, 3)
        self.u1 = un1
        self.u2 = un2
        self.u3 = un3

    def heat_equation(self, u, north, south, west, east, room):
        un = np.zeros((len(u[:,0]), len(u[0,:])))
        #Interior
        un[1:-1,1:-1] = (u[2:,1:-1] + u[:-2,1:-1] - 4*u[1:-1,1:-1] + u[1:-1,2:] + u[1:-1,:-2])
        #Walls
        un[0,1:-1] = (north[1:-1] + u[1,1:-1] -4*u[0,1:-1] + u[0,0:-2] + u[0,2:])
        un[-1,1:-1] = (south[1:-1] + u[-2,1:-1] -4* u[-1,1:-1] + u[-1,0:-2] + u[-1,2:])
        if room == 3:
            un[1:-1,0] = (u[1:-1,1] -3* u[1:-1,0] + u[0:-2,0] + u[2:,0]) #West wall
            un[0,0] = (north[0] - 3*u[0,0] + u[0,1] + u[1,0])
            un[-1,0] = (south[0] - 3*u[-1,0] + u[-2,0] + u[-1,1])
        else:
            un[1:-1,0] = (west[1:-1] + u[1:-1,1] -4* u[1:-1,0] + u[0:-2,0] + u[2:,0]) #West wall
            un[0,0] = (north[0] + west[0] - 4*u[0,0] + u[0,1] + u[1,0])
            un[-1,0] = (south[0] + west[-1] - 4*u[-1,0] + u[-2,0] + u[-1,1])
        if room == 1:
            un[1:-1,-1] = (u[1:-1,-2] -3* u[1:-1,-1] + u[0:-2,-1] + u[2:,-1]) #East wall
            un[0,-1] = (north[-1] - 3*u[0,-1] + u[0,-2] + u[1,-1])
            un[-1,-1] = (south[-1] - 3*u[-1,-1] + u[-1,-2] + u[-2,-1])
        else:
            un[1:-1,-1] = (east[1:-1] + u[1:-1,-2] -4* u[1:-1,-1] + u[0:-2,-1] + u[2:,-1])
            un[0,-1] = (north[-1] + east[0] - 4*u[0,-1] + u[0,-2] + u[1,-1])
            un[-1,-1] = (south[-1] + east[-1] - 4*u[-1,-1] + u[-1,-2] + u[-2,-1])
        return un * self.D + u


AP = Appartment(delta_x = 1/100, D = 0.2)
U = np.ones((2*AP.n, 3*AP.n)) * AP.u2.min()
U[AP.n:, :AP.n] = AP.u1
U[:, AP.n:2*AP.n] = AP.u2
U[:AP.n, 2*AP.n:] = AP.u3
print((AP.u1.sum() +  AP.u2.sum() + AP.u3.sum())/(4*AP.n**2))
for i in range(10000):
    AP.step()
U = np.ones((2*AP.n, 3*AP.n)) * (AP.u2.min()+AP.u2.max())/2
U[AP.n:, :AP.n] = AP.u1
U[:, AP.n:2*AP.n] = AP.u2
U[:AP.n, 2*AP.n:] = AP.u3
print('Average =', (AP.u1.sum() +  AP.u2.sum() + AP.u3.sum())/(4*AP.n**2), 'Min =', U.min(), 'Max = ', U.max())
#plt.imshow(U, cmap = "seismic")
plt.imshow(U, cmap = "twilight_shifted")
plt.colorbar()
plt.show()