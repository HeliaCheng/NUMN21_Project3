#Lagrangian solver in 2d, for rectangular geometry.
#The boundary condition can be set on each side.
#Only Dirichlet is set for now

import numpy as np
import matplotlib.pyplot as plt



def create_lagrangian_matrix(boundary,h):
    """
    boundary is a dictionary with all boundaries
    

    Does support Neumann condtion, but need to be tested further.
     
    Can be changed if necessary.
    

    """
    
    left, left_cdt = boundary["left"]
    right, right_cdt = boundary["right"]
    bottom, bottom_cdt = boundary["bottom"]
    top, top_cdt = boundary["top"]
    
    assert np.shape(left) == np.shape(right)
    assert np.shape(top) == np.shape(bottom)
    n = len(left)
    m = len(bottom)
    nm = n*m

    A = np.zeros((nm,nm))
    #print(A.shape)
    #We work from row to row, column to column
    #First, interior points
    for row in range(n):
        for col in range(m):
          A_row = np.zeros(nm)
          center = row*m + col
          A_row[center] = -4 #Assume Dirichlet bc by default
          #Try to fill every point, if we are out of bound,
          #then add the boundary condition
          if(col != m-1):
            A_row[center +1] = 1 #Point to the right
          else:
            if(right_cdt.lower() == "neumann"):
                A_row[center] += 1
            
          if(col != 0):
            A_row[center -1] = 1 #Point to the left
          else:
            if(left_cdt.lower() == "neumann"):
                A_row[center] += 1
          if(row != n-1):
            A_row[center +m] = 1 #Point below
          else:
            if(bottom_cdt.lower() == "neumann"):
                A_row[center] += 1
          if(row != 0):
            A_row[center -m] = 1 #Point above
          else:
            if(top_cdt.lower() == "neumann"):
                A_row[center] += 1
          A[center,:] = A_row
    A *= 1/h**2
    return A





def create_lagrangian_rhs(boundary,h):
    left, left_cdt = boundary["left"]
    right, right_cdt = boundary["right"]
    top , top_cdt = boundary["top"]
    bottom , bottom_cdt = boundary["bottom"]
    n = np.shape(left)[0]
    m = np.shape(top)[0]
    mn = m*n
    assert n == np.shape(right)[0]
    assert m == np.shape(bottom)[0]
    
    b_matrix = np.zeros((n,m))
    
    #Rule one of programming is to repeat yourself.
    if(top_cdt.lower() == "neumann"):
        b_matrix[0,:] -= top / h
    elif(top_cdt.lower() == "dirichlet"):
        b_matrix[0,:] -= top / h**2
    else: 
        raise ValueError("Unknown boundary condition")
    
    if(bottom_cdt.lower() == "neumann"):
        b_matrix[n-1,:] -= bottom / h
    elif(bottom_cdt.lower() == "dirichlet"):
        b_matrix[n-1,:] -= bottom / h**2
    else: 
        raise ValueError("Unknown boundary condition")
      
    if(left_cdt.lower() == "neumann"):
        b_matrix[:,0] -= left / h
    elif(left_cdt.lower() == "dirichlet"):
        b_matrix[:,0] -= left / h**2
    else: 
        raise ValueError("Unknown boundary condition")
    
    if(right_cdt.lower() == "neumann"):
        b_matrix[:,m-1] -= right / h
    elif(right_cdt.lower() == "dirichlet"):
        b_matrix[:,m-1] -= right / h**2
    else: 
        raise ValueError("Unknown boundary condition")
    
    return b_matrix.flatten()



def visualize_stencil(A,i,j,n,m, h=1):
    """Visualize the stencil used, 

    Args:
        A (_type_): Matrix to visualize
        m (_type_): length of the left/right border
        n (_type_): length of the top/bottom border
        i (_type_): row to visualize
        j (_type_): column to visualize
    """
    A_row = A[i * n + j,:]
    stencil = A_row.reshape((n,m)) * h**2
    fig, ax = plt.subplots()
    img = ax.imshow(stencil)
    fig.colorbar(img)
    plt.show()
    

def solve_lagrangian(A,b,n,m):
    v =  np.linalg.solve(A,b)
    return v.reshape((n,m))




if __name__ == "__main__":

    n = 5
    m = 6
    left = np.ones(n) * 10
    right = np.ones(n) * 1

    top = np.ones(m) * 5
    bottom = np.ones(m) * 5

    h = 0.5



    boundary = {"left": [left,"neumann"],
                "right": [right,"dirichlet"],
                "top": [top,"dirichlet"],
                "bottom": [bottom,"dirichlet"]}


    A =  create_lagrangian_matrix(boundary,h)
    b = create_lagrangian_rhs(boundary,h)
    


    v = solve_lagrangian(A,b,n,m)
    
    plt.imshow(v)
    plt.show()




class Apartment:
    def __init__(self, delta_x=1/20, initial_temp=20, D=0.1):
        self.delta_x = delta_x
        self.D = D
        self.initial_temp = initial_temp
        self.omega = 0.8
        self.n = int(1 / delta_x)  # Grid points per unit length
        self.u1 = np.ones((self.n, self.n)) * self.initial_temp  # Room 1
        self.u2 = np.ones((2 * self.n, self.n)) * self.initial_temp  # Room 2
        self.u3 = np.ones((self.n, self.n)) * self.initial_temp  # Room 3
        
        self.H = 40  # Heated boundary
        self.WF = 5  # Window (cold boundary)
        self.NW = 15  # Normal wall
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

        
    def update_walls(self):
        self.east_wall_1 = self.u2[self.n:,0]
        self.west_wall_2[self.n:] = self.u1[:,-1]
        self.east_wall_2[:self.n] = self.u3[:,0]
        self.west_wall_3 = self.u2[:self.n,-1]
        
    def create_matrix_and_rhs(self, u, north, south, west, east):
        # Define boundary conditions for each room
        if u == 1:
            boundary = {
                "left": (west, "dirichlet"),
                "right": (east, "neumann"),
                "bottom": (south, "dirichlet"),
                "top": (north, "dirichlet")
            }
        elif u == 3:
            boundary = {
                "left": (west, "neumann"),
                "right": (east, "dirichlet"),
                "bottom": (south, "dirichlet"),
                "top": (north, "dirichlet")
            }
        else:
            boundary = {
                "left": (west, "dirichlet"),
                "right": (east, "dirichlet"),
                "bottom": (south, "dirichlet"),
                "top": (north, "dirichlet")
            }
        A = create_lagrangian_matrix(boundary, self.delta_x)
        b = create_lagrangian_rhs(boundary, self.delta_x)
        return A, b
    
    def step(self):
        # Update boundary walls
        self.update_walls()
        #Solve the room with Dirichlet conditions first, then send the Neumann cdt to the other ones.

        # Room 2 - the big one
        A2, b2 = self.create_matrix_and_rhs(2, self.north_wall_2, self.south_wall_2, self.west_wall_2, self.east_wall_2)
        un2 = solve_lagrangian(A2, b2, 2*self.n, self.n)

        east_wall1_neumann = -(self.u2[self.n:,0]-self.u2[self.n:,1])/self.delta_x
        west_wall3_neumann = -(self.u2[:self.n,-1] - self.u2[:self.n,-2])/self.delta_x
        # Room 1 - left one
        A1, b1 = self.create_matrix_and_rhs(1, self.north_wall_1, self.south_wall_1, self.west_wall_1, east_wall1_neumann)
        un1 = solve_lagrangian(A1, b1, self.n, self.n)
        

        
        # Room 3 - right one
        A3, b3 = self.create_matrix_and_rhs(3, self.north_wall_3, self.south_wall_3, west_wall3_neumann, self.east_wall_3)
        un3 = solve_lagrangian(A3, b3, self.n, self.n)
        
        # Update room temperatures
        self.u1 = self.omega*un1 + (1-self.omega)*self.u1
        self.u2 = self.omega*un2 + (1-self.omega)*self.u2
        self.u3 = self.omega*un3 + (1-self.omega)*self.u3

    def visualize(self):
        U = np.ones((2 * self.n, 3 * self.n)) * (self.u2.min() + self.u2.max()) / 2
        U[self.n:, :self.n] = self.u1
        U[:, self.n:2 * self.n] = self.u2
        U[:self.n, 2 * self.n:] = self.u3
        plt.imshow(U, cmap="twilight_shifted")
        plt.colorbar()
        plt.show()

# Instantiate and run
apartment = Apartment(delta_x=1/40, D=0.2)
apartment.omega = 0.8
for i in range(20):
    apartment.step()

apartment.visualize()





class Apartment3a:
    def __init__(self, delta_x=1/20, initial_temp=20, D=0.1, rooms = 3):
        self.delta_x = delta_x
        self.D = D
        self.initial_temp = initial_temp
        self.omega = 0.8
        self.rooms = rooms
        self.n = int(1 / delta_x)  # Grid points per unit length
        self.u1 = np.ones((self.n, self.n)) * self.initial_temp  # Room 1
        self.u2 = np.ones((2 * self.n, self.n)) * self.initial_temp  # Room 2
        self.u3 = np.ones((self.n, self.n)) * self.initial_temp  # Room 3
        
        self.H = 40  # Heated boundary
        self.WF = 5  # Window (cold boundary)
        self.NW = 15  # Normal wall
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
        self.west_wall_3 = np.ones(int(1/delta_x)) * self.NW  #Should depend on room 2
        if self.rooms == 4:
            self.u4 = np.ones((int(self.n/2), int(self.n/2))) * self.initial_temp
            self.north_wall_4 = np.ones(int(0.5/delta_x)) * self.NW #Should depend on room 3
            self.east_wall_4 = np.ones(int(0.5/delta_x)) * self.NW
            self.south_wall_4 = np.ones(int(0.5/delta_x)) * self.H
            self.west_wall_4 = np.ones(int(0.5/delta_x)) * self.NW  #Should depend on room 2
        self.boundaries()
        self.init_A()

        
    def update_walls(self):
        self.east_wall_1 = self.u2[self.n:,0]
        self.west_wall_2[self.n:] = self.u1[:,-1]
        self.east_wall_2[:self.n] = self.u3[:,0]
        self.west_wall_3 = self.u2[:self.n,-1]
        if self.rooms == 4:
            self.south_wall_3[:int(self.n*0.5)] = self.u4[0,:]
            self.east_wall_2[self.n:int(self.n*1.5)] = self.u4[:,0]
        
    def boundaries(self):
        self.boundary1 = {
            "left": (self.west_wall_1, "dirichlet"),
            "right": (self.east_wall_1, "neumann"),
            "bottom": (self.south_wall_1, "dirichlet"),
            "top": (self.north_wall_1, "dirichlet")
        }
        self.boundary2 = {
            "left": (self.west_wall_2, "dirichlet"),
            "right": (self.east_wall_2, "dirichlet"),
            "bottom": (self.south_wall_2, "dirichlet"),
            "top": (self.north_wall_2, "dirichlet")
        }
        self.boundary3 = {
            "left": (self.west_wall_3, "neumann"),
            "right": (self.east_wall_3, "dirichlet"),
            "bottom": (self.south_wall_3, "dirichlet"),
            "top": (self.north_wall_3, "dirichlet")
        }
        if self.rooms == 4:
            self.boundary4 = {
                "left": (self.west_wall_4, "neumann"),
                "right": (self.east_wall_4, "dirichlet"),
                "bottom": (self.south_wall_4, "dirichlet"),
                "top": (self.north_wall_4, "neumann")
            }

    def init_A(self):
        self.A1 = create_lagrangian_matrix(self.boundary1, self.delta_x)
        self.A2 = create_lagrangian_matrix(self.boundary2, self.delta_x)
        self.A3 = create_lagrangian_matrix(self.boundary3, self.delta_x)
        if self.rooms == 4:
            self.A4 = create_lagrangian_matrix(self.boundary4, self.delta_x)

    def update_b(self):
        self.b1 = create_lagrangian_rhs(self.boundary1, self.delta_x)
        self.b2 = create_lagrangian_rhs(self.boundary2, self.delta_x)
        self.b3 = create_lagrangian_rhs(self.boundary3, self.delta_x)
        if self.rooms == 4:
            self.b4 = create_lagrangian_rhs(self.boundary4, self.delta_x)

    def step(self):
        # Update boundary walls
        self.update_walls()
        self.east_wall_1 = -(self.u2[self.n:,0]-self.u2[self.n:,1])/self.delta_x
        self.west_wall_3 = -(self.u2[:self.n,-1] - self.u2[:self.n,-2])/self.delta_x
        if self.rooms == 4:
            self.west_wall_4 = (self.u2[self.n:int(self.n*1.5), -1] - self.u2[self.n:int(self.n*1.5),-2])/self.delta_x
            self.north_wall_4 = -(self.u3[-1, :int(self.n*0.5)] - self.u3[-2, :int(self.n*0.5)])/self.delta_x
        
        self.boundaries()
        self.update_b()
        
        #Solve the room with Dirichlet conditions first, then send the Neumann cdt to the other ones.
        # Room 2 - the big one
        un2 = solve_lagrangian(self.A2, self.b2, 2*self.n, self.n)

        
        # Room 1 - left one
        un1 = solve_lagrangian(self.A1, self.b1, self.n, self.n)
        
        # Room 3 - right one
        un3 = solve_lagrangian(self.A3, self.b3, self.n, self.n)

        if self.rooms == 4:
            # Room 4 - small one
            un4 = solve_lagrangian(self.A4, self.b4, int(0.5*self.n), int(0.5*self.n))
            self.u4 = self.omega*un4 + (1-self.omega)*self.u4
        
        # Update room temperatures
        self.u1 = self.omega*un1 + (1-self.omega)*self.u1
        self.u2 = self.omega*un2 + (1-self.omega)*self.u2
        self.u3 = self.omega*un3 + (1-self.omega)*self.u3

        

    def visualize(self):
        U = np.ones((2 * self.n, 3 * self.n)) * (self.u2.min() + self.u2.max()) / 2
        U[self.n:, :self.n] = self.u1
        U[:, self.n:2 * self.n] = self.u2
        U[:self.n, 2 * self.n:] = self.u3
        if self.rooms == 4:
            U[self.n:int(self.n*1.5), self.n*2:int(self.n*2.5)] = self.u4
        plt.imshow(U, cmap="twilight_shifted")
        plt.colorbar()
        plt.show()
        av1 = self.u1.sum()/(self.n**2)
        av2 = self.u2.sum()/(2*self.n**2)
        av3 = self.u3.sum()/(self.n**2)
        if self.rooms == 4:
            av4 = self.u4.sum()/(0.25*self.n**2)
            av_ap = (av1 + 2*av2 + av3 + av4*0.25)/4.25
        else:
            av_ap = (av1 + 2*av2 + av3)/4
        print('Average room 1:', av1)
        print('Average room 2:', av2)
        print('Average room 3:', av3)
        if self.rooms == 4:
            print('Average room 4:', av4)
        print('Average appartment:', av_ap)

# Instantiate and run
apartment3a = Apartment3a(delta_x=1/40, D=0.2, rooms = 4)
apartment3a.omega = 0.8
for i in range(10):
    apartment3a.step()

apartment3a.visualize()
