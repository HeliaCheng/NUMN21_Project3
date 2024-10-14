
from mpi4py import MPI
import numpy as np
from lagrangian_solver import solve_lagrangian,visualize_stencil,create_lagrangian_rhs,create_lagrangian_matrix,DN_solver
import MPI
from ApartmentSetting import get_room_list,get_config_by_rank,Room

def main():
    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    room_list=get_room_list()

    solver =DN_solver(h=1 / 20, initial_temp=20, D=0.1,room_list=room_list)
    u_new = solve_lagrangian()



    # Terminate MPI
    MPI.Finalize()


if __name__ == "__main__":
    main()