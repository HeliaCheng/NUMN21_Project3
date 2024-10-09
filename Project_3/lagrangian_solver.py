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
    print(A.shape)
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
