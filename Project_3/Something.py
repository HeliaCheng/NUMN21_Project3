import numpy as np
from mpi4py import MPI

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

    def heat_equation(self):
        """
        Implements heat equation
        """
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