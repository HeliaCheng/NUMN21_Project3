import numpy as np
from mpi4py import MPI

class Appartment(): #Maybe break up into different classes. Give better name?
    def __init__(self, delta_x = 1/20):
        self.delta_x = delta_x
        self.initial_temp = 20
        self.u1 = np.ones((int(1/delta_x), int(1/delta_x))) * self.initial_temp #the different rooms, 
        self.u2 = np.ones((int(1/delta_x), int(2/delta_x))) * self.initial_temp
        self.u3 = np.ones((int(1/delta_x), int(1/delta_x))) * self.initial_temp
        self.NW = 15 #heat normal wall
        self.WH = 40 #heat walls with heater
        self.WW = 5  #heat wall with bit window
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
    def __init__(self) -> None:
        pass

    def define_boundary(self):
        pass

class Boundary():
    def __init__(self) -> None:
        self.H = np.zeros(20)*40
        pass