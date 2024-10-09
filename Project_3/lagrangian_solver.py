#Lagrangian solver in 2d, for rectangular geometry.
#The boundary condition can be set on each side.
#Only Dirichlet is set for now

import numpy as np
import matplotlib.pyplot as plt



def lagrangian_problem(left,right,top,bottom,h):
    """
    left,right,top,bottom are vector with the boudnary condition

    If Neumann, add ghost cells, should work out quite well.

    """
    assert np.shape(left) == np.shape(right)
    assert np.shape(top) == np.shape(bottom)
    n = len(left)
    m = len(bottom)
    nm = n*m
    left_cdt = "dirichlet"
    right_cdt = "dirichlet"
    bottom_cdt = "dirichlet"
    top_cdt = "dirichlet"
    A = np.zeros((nm,nm))
    b = np.zeros(nm) #Data vector, we will fill it

    #We work from row to row, column to column
    #First, interior points
    for row in range(n):
        for col in range(m):
          A_row = np.zeros(nm)
          center = row*m + col
          A_row[center] = -4
          #Try to fill every point, if we are out of bound,
          #then add the boundary condition
          if(col != m-1):
            A_row[row*m + col +1] = 1 #Point to the right
          else:
            b[row*m+col] -= right[row]/h**2
          if(col != 0):
            A_row[row*m + col -1] = 1 #Point to the left
          else:
            b[row*m+col] -= left[row]/h**2
          if(row != n-1):
            A_row[row*m + col +m] = 1 #Point below
          else:
            b[row*m+col] -= bottom[col]/h**2
          if(row != 0):
            A_row[row*m + col -m] = 1 #Point above
          else:
            b[row*m+col] -= top[col]/h**2
          A[row*m+col,:] = A_row
    A *= 1/h**2
    return A , b , n , m



if __name__ == "__main__":

    left = np.ones(20) * 10
    right = np.ones(20) * 1

    top = np.ones(30) * 5
    bottom = np.ones(30) * 5

    h = 1

    A,b, n, m = lagrangian_problem(left,right,top,bottom,h)


    print(A)
    print(b)

    def room_solve(A,b,n,m):
        v =  np.linalg.solve(A,b)
        return v.reshape((n,m))

    v = room_solve(A,b,n,m)
    print(v)

