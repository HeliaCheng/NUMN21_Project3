import numpy as np

class Appartment(): #Maybe break up into different classes. Give better name?
    def __init__(self, delta_x = 1/20):
        self.delta_x = delta_x
        self.initial_temp = 20
        self.u01 = np.ones((int(1/delta_x), int(1/delta_x))) * self.initial_temp #the different rooms, 
        self.u02 = np.ones((int(1/delta_x), int(2/delta_x))) * self.initial_temp
        self.u03 = np.ones((int(1/delta_x), int(1/delta_x))) * self.initial_temp
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

    def neumann(self):
        """
        Implements Neumann boundary conditions
        """
        pass

    def dirichlet(self):
        """
        Implements Neumann boundary conditions
        """
        pass
