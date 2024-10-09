import numpy as np
from mpi4py import MPI
from scipy import sparse
from scipy.sparse.linalg import spsolve

class Appartment(): #Maybe break up into different classes. Give better name?
    def __init__(self, delta_x = 1/20):
        self.delta_x = delta_x
        self.omega = 0.8 #Used in relaxation   


    def step(self):
        """
        Does one time step
        """
        pass

    def relaxation(self):
        """
        Implements the relaxation
        """
        pass

    def heat_equation(self, nx, nt, dx, dt, alpha):
        """
        Implements heat equation

        Args: 
            nx: Number of spatial points
            nt: Number of time steps
            dx:
            dt:
            alpha:  Thermal diffusivity
         """

        #Create sparse matrix
        diagonals = [-alpha*dt/dx**2, 1+2*alpha*dt/dx**2, -alpha*dt/dx**2]
        offsets = [-1, 0, 1]
        A = sparse.diags(diagonals, offsets, shape=(nx, nx), format='csc')[3]

        #Set up Initial and Boundary condition 
        u = np.zeros(nx)
        u[0] = 0  # Left boundary
        u[-1] = 1  # Right boundary

        #Solve the system for each time step
        for _ in range(nt):
            b = u.copy()
            b[0] = 0  # Left boundary
            b[-1] = 1  # Right boundary
            u = spsolve(A, b)

        pass

    def interior(self, room):
        """
        Input: all points in a room
        Output: all points not next to a boundary updated
        """
        pass

    def wall_boundary(self, air, heat): 
        """
        Input: two lines of air next to the wall, heat of gthe wall
        Output: updated air
        """
        pass

    def neumann(self):
        pass

    def dirichlet(self):
        pass

    def neumann_dirichlet(self, dir_side, neu_side):
        """
        Input: sides is the air on each side of the boundary
        Output: updated air
        """
        pass


class Room():
    def __init__(self, delta_x = 1/20, initial_temp = 20):
        self.delta_x = delta_x
        self.initial_temp = initial_temp
        self.n = int(1/delta_x)
        self.u1 = np.ones((self.n, self.n)) * self.initial_temp #the different rooms at t = 0 
        self.u2 = np.ones((self.n, 2*self.n)) * self.initial_temp
        self.u3 = np.ones((self.n, self.n)) * self.initial_temp
        pass

    def define_boundary(self):
        pass

class Boundary():
    def __init__(self, delta_x = 1/20) -> None:
        self.H = 40 #Heated
        self.WF = 5 #Window
        self.NW = 15#Normal
        self.delta_x = delta_x
        #Walls room 1:
        self.north_wall_1 = np.ones(int(1/delta_x)) * self.NW
        self.east_wall_1 = np.ones(int(1/delta_x)) * self.NW  #Should de pend on room 2
        self.south_wall_1 = np.ones(int(1/delta_x)) * self.NW
        self.west_wall_1 = np.ones(int(1/delta_x)) * self.H
        #Walls room 2:
        self.north_wall_2 = np.ones(int(1/delta_x)) * self.H
        self.east_wall_2 = np.ones(int(2/delta_x)) * self.NW #Should partially depend on room 3
        self.south_wall_2 = np.ones(int(1/delta_x)) * self.WF
        self.west_wall_2 = np.ones(int(2/delta_x)) * self.NW #Should partially depend on room 1
        #Walls room 3:
        self.north_wall_3 = np.ones(int(1/delta_x)) * self.H
        self.east_wall_3 = np.ones(int(1/delta_x)) * self.H
        self.south_wall_3 = np.ones(int(1/delta_x)) * self.H
        self.west_wall_3 = np.ones(int(1/delta_x)) * self.H  #Should de pend on room 2
        pass