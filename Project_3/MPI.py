# Lagrangian solver in 2d, for rectangular geometry.
# The boundary condition can be set on each side.
# Only Dirichlet is set for now

import numpy as np
import matplotlib.pyplot as plt


def create_lagrangian_matrix(room, h):
    """
    boundary is a dictionary with all boundaries
    Does spport Neumann condtion, but need to be tested further.
    Can be changed if necessary.
    """
    dim=room.dim
    left_b = room.left_b
    right_b= room.right_b
    bottom_b = room.bottom_b
    top_b = room.top_b

    [n,m] = dim/h#
    nm = n * m

    A = np.zeros((nm, nm))

    for row in range(n):
        for col in range(m):
            A_row = np.zeros(nm)
            center = row * m + col
            A_row[center] = -4  # Assume Dirichlet bc by default
            # Try to fill every point, if we are out of bound,
            # then add the boundary condition
            if (col != m - 1):
                A_row[center + 1] = 1  # Point to the right
            else:
                if (right_b.lower() == "neumann"):
                    A_row[center] += 1

            if (col != 0):
                A_row[center - 1] = 1  # Point to the left
            else:
                if (left_b.lower() == "neumann"):
                    A_row[center] += 1
            if (row != n - 1):
                A_row[center + m] = 1  # Point below
            else:
                if (bottom_b.lower() == "neumann"):
                    A_row[center] += 1
            if (row != 0):
                A_row[center - m] = 1  # Point above
            else:
                if (top_b.lower() == "neumann"):
                    A_row[center] += 1
            A[center, :] = A_row
    A *= 1/h ** 2
    return A


def create_lagrangian_rhs(room, h):
    dim=room.dim
    left_b = room.left_b
    right_b = room.right_b
    top_b = room.top_b
    bottom_b = room.bottom_b
    [m,n]=dim*h
    mn = m * n

    b_matrix = np.zeros((n, m))

    # Rule one of programming is to repeat yourself.
    if (top_b.lower() == "neumann"):
        b_matrix[0, :] -= np.ones(int(1/h)) * room.top_t / h
    elif (top_b.lower() == "dirichlet"):
        b_matrix[0, :] -= np.ones(int(1/h)) * room.top_t / h ** 2
    else:
        raise ValueError("Unknown boundary condition")

    if (bottom_b.lower() == "neumann"):
        b_matrix[n - 1, :] -= np.ones(int(1/h)) * room.bottom_t / h
    elif (bottom_b.lower() == "dirichlet"):
        b_matrix[n - 1, :] -= np.ones(int(1/h)) * room.bottom_t  / h ** 2
    else:
        raise ValueError("Unknown boundary condition")

    if (left_b.lower() == "neumann"):
        b_matrix[:, 0] -= np.ones(int(1/h)) * room.left_t  / h
    elif (left_b.lower() == "dirichlet"):
        b_matrix[:, 0] -= np.ones(int(1/h)) * room.left_t  / h ** 2
    else:
        raise ValueError("Unknown boundary condition")

    if (right_b.lower() == "neumann"):
        b_matrix[:, m - 1] -= np.ones(int(1/h)) * room.right_t / h
    elif (right_b.lower() == "dirichlet"):
        b_matrix[:, m - 1] -= np.ones(int(1/h)) * room.right_t  / h ** 2
    else:
        raise ValueError("Unknown boundary condition")

    return b_matrix.flatten()

def create_matrix_and_rhs(room,h):
        A = create_lagrangian_matrix(room, h)
        b = create_lagrangian_rhs(room, h)
        return A, b
def visualize_stencil(A, i, j, n, m, h=1):
    """Visualize the stencil used,

    Args:
        A (_type_): Matrix to visualize
        m (_type_): length of the left/right border
        n (_type_): length of the top/bottom border
        i (_type_): row to visualize
        j (_type_): column to visualize
    """
    A_row = A[i * n + j, :]
    stencil = A_row.reshape((n, m)) * h ** 2
    fig, ax = plt.subplots()
    img = ax.imshow(stencil)
    fig.colorbar(img)
    plt.show()


def solve_lagrangian(A, b, n, m):
    v = np.linalg.solve(A, b)
    return v.reshape((n, m))



class MPI:
    def __init__(self, h=1 / 20, initial_temp=20, D=0.1,room_list=[]):
        self.h = h
        self.D = D
        self.initial_temp = initial_temp
        self.omega = 0.8
        self.room_list=room_list


#********************************************************************************************
    def update_walls(self):
        self.east_wall_1 = self.u2[self.n:, 0]
        self.west_wall_2[self.n:] = self.u1[:, -1]
        self.east_wall_2[:self.n] = self.u3[:, 0]
        self.west_wall_3 = self.u2[:self.n, -1]


#*********************************************************************************************
    def step(self):
        # Update boundary walls
        self.update_walls()
        # Solve the room with Dirichlet conditions first, then send the Neumann cdt to the other ones.

        # Room 2 - the big one
        A2, b2 = self.create_matrix_and_rhs(2, self.north_wall_2, self.south_wall_2, self.west_wall_2, self.east_wall_2)
        un2 = solve_lagrangian(A2, b2, 2 * self.n, self.n)

        east_wall1_neumann = -(self.u2[self.n:, 0] - self.u2[self.n:, 1]) / self.delta_x
        west_wall3_neumann = -(self.u2[:self.n, -1] - self.u2[:self.n, -2]) / self.delta_x
        # Room 1 - left one
        A1, b1 = self.create_matrix_and_rhs(1, self.north_wall_1, self.south_wall_1, self.west_wall_1,
                                            east_wall1_neumann)
        un1 = solve_lagrangian(A1, b1, self.n, self.n)

        # Room 3 - right one
        A3, b3 = self.create_matrix_and_rhs(3, self.north_wall_3, self.south_wall_3, west_wall3_neumann,
                                            self.east_wall_3)
        un3 = solve_lagrangian(A3, b3, self.n, self.n)

        # Update room temperatures
        self.u1 = self.omega * un1 + (1 - self.omega) * self.u1
        self.u2 = self.omega * un2 + (1 - self.omega) * self.u2
        self.u3 = self.omega * un3 + (1 - self.omega) * self.u3

    def visualize(self):
        U = np.ones((2 * self.n, 3 * self.n)) * (self.u2.min() + self.u2.max()) / 2
        U[self.n:, :self.n] = self.u1
        U[:, self.n:2 * self.n] = self.u2
        U[:self.n, 2 * self.n:] = self.u3
        plt.imshow(U, cmap="twilight_shifted")
        plt.colorbar()
        plt.show()


# Instantiate and run
MPI = MPI(delta_x=1 / 40, D=0.2)
MPI.omega = 0.8
for i in range(20):
    MPI.step()

MPI.visualize()
