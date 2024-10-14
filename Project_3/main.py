
from mpi4py import MPI
import numpy as np
from lagrangian_solver import solve_lagrangian,visualize_stencil,create_lagrangian_rhs,create_lagrangian_matrix
import Apartment
def main():
    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Solve
    rooms=Apartment(delta_x = 1/100, D = 0.2)
    solver =
    u_new = solve_lagrangian()



    # Terminate MPI
    MPI.Finalize()


if __name__ == "__main__":
    main()