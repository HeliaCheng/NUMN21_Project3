from matplotlib.patches import Rectangle
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
# from scipy.sparse import csr_array
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from ApartmentSetting import get_room_list,get_config_by_rank,Room

start_time = time.time()
def create_lagrangian_matrix(room, h):
    """
    boundary is a dictionary with all boundaries


    Does support Neumann condtion, but need to be tested further.

    Can be changed if necessary.


    """
    dim=room.dim
    [n, m] = [int(dim[0] / h), int(dim[1] / h)]
    nm = n * m
    rows = []
    cols = []
    values = []

    def add_to_lists(row, col, value):
        """Add the value of the matrix at the specified row and column"""
        rows.append(row)
        cols.append(col)
        values.append(value)

    # print(A.shape)
    # We work from row to row, column to column
    # First, interior points
    for row in range(n):
        for col in range(m):

            center = row * m + col
            add_to_lists(center, center, -4)
            # Try to fill every point, if we are not out of bound
            if (col != m - 1):  # Point to the right
                add_to_lists(center, center + 1, 1)
            elif (room.right_b.lower() == "neumann"):
                add_to_lists(center, center, 1)
            if (col != 0):  # Point to the left
                add_to_lists(center, center - 1, 1)
            elif (room.left_b.lower() == "neumann"):
                add_to_lists(center, center, 1)
            if (row != n - 1):  # Point below
                add_to_lists(center, center + m, 1)
            elif (room.bottom_b.lower() == "neumann"):
                add_to_lists(center, center, 1)
            if (row != 0):  # Point above
                add_to_lists(center, center - m, 1)
            elif (room.top_b.lower() == "neumann"):
                add_to_lists(center, center, 1)
    return csr_matrix((values, (rows, cols)), shape=(nm, nm)) / h ** 2
# def create_lagrangian_matrix(room, h):
#     """
#     boundary is a dictionary with all boundaries
#     Does spport Neumann condtion, but need to be tested further.
#     Can be changed if necessary.
#     """
#     dim=room.dim
#     left_b = room.left_b
#     right_b= room.right_b
#     bottom_b = room.bottom_b
#     top_b = room.top_b
#
#     [n,m] = [int(dim[0]/h),int(dim[1]/h)]#
#     nm = n * m
#
#     A = np.zeros((nm, nm))
#
#     for row in range(n):
#         for col in range(m):
#             A_row = np.zeros(nm)
#             center = row * m + col
#             A_row[center] = -4  # Assume Dirichlet bc by default
#             # Try to fill every point, if we are out of bound,
#             # then add the boundary condition
#             if (col != m - 1):
#                 A_row[center + 1] = 1  # Point to the right
#             else:
#                 if (right_b.lower() == "neumann"):
#                     A_row[center] += 1
#
#             if (col != 0):
#                 A_row[center - 1] = 1  # Point to the left
#             else:
#                 if (left_b.lower() == "neumann"):
#                     A_row[center] += 1
#             if (row != n - 1):
#                 A_row[center + m] = 1  # Point below
#             else:
#                 if (bottom_b.lower() == "neumann"):
#                     A_row[center] += 1
#             if (row != 0):
#                 A_row[center - m] = 1  # Point above
#             else:
#                 if (top_b.lower() == "neumann"):
#                     A_row[center] += 1
#             A[center, :] = A_row
#     A *= 1/h ** 2
#     return A


def create_lagrangian_rhs(room, h):
    dim=room.dim

    [n,m]=[int(dim[0]/h),int(dim[1]/h)]
    mn = m * n

    b_matrix = np.zeros((n, m))

    # Rule one of programming is to repeat yourself.
    if (room.top_b.lower() == "neumann"):
        b_matrix[0, :] -= room.wall_r_top.flatten()/ h
    elif (room.top_b.lower() == "dirichlet"):
        b_matrix[0, :] -= room.wall_r_top.flatten()/ h ** 2
    else:
        raise ValueError("Unknown boundary condition")

    if (room.bottom_b.lower() == "neumann"):
        b_matrix[n- 1, :] += room.wall_r_bottom.flatten() / h
    elif (room.bottom_b.lower() == "dirichlet"):
        b_matrix[n - 1, :] += room.wall_r_bottom.flatten() / h ** 2
    else:
        raise ValueError("Unknown boundary condition")

    if (room.left_b.lower() == "neumann"):
        b_matrix[:, 0] +=room.wall_r_left.flatten() / h
    elif (room.left_b.lower() == "dirichlet"):
        b_matrix[:, 0] += room.wall_r_left.flatten() / h ** 2
    else:
        raise ValueError("Unknown boundary condition")

    if (room.right_b.lower() == "neumann"):
        b_matrix[:, m - 1] -= room.wall_r_right.flatten()/ h
    elif (room.right_b.lower() == "dirichlet"):
        b_matrix[:, m - 1] -= room.wall_r_right.flatten()/ h ** 2
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
    v =  spsolve(A,b)
    return v.reshape((n, m))


class DN_solver:
    def __init__(self, h, initial_temp,room_list,comm,rank,size):
        self.comm=comm
        self.rank=rank
        self.h = h
        self.n=int(1/h)
        self.initial_temp = initial_temp
        self.omega = 0.8
        self.room_list=room_list
        self.rooms=4
        self.iter=10
        self.size=size
        # self.u_dict = {i: room.u for i, room in enumerate(self.room_list)}
#python would adjust the original list when adjust the list with new reference
    #aka: the adjustments in u_dict would be the same as of room.u
#*********************************************************************************************
    # def step(self):
    #     print(f"Hello from process {self.rank} out of {self.size} processes")
    #     self.comm.Barrier()#useless,why
    #
    #     for k in range(self.iter):
    #
    #         print("iteration : *********************",k,"rank***********",self.rank)
    #         comm.Barrier()
    #         u_dict = {i: room.u for i, room in enumerate(self.room_list)}
    #         for i, room in enumerate(self.room_list):
    #
    #             print(self.rank,"self.rank")
    #             # print("barrier",self.rank)
    #             # self.comm.Barrier()
    #             if self.rank == i:
    #                  print("Now room ", i)
    #                  u = u_dict[i]
    #                  adjacent_rooms = room.adjacent_rooms
    #                  # Send,Receive,Update
    #
    #                  self.send_conditions(u, adjacent_rooms)
    #                  print("send complete")
    #
    #                  u = self.receive_and_update_conditions(u, adjacent_rooms)
    #                  print("receive complete")
    #                  # Solve
    #                  A,b=create_matrix_and_rhs(room,self.h)
    #                  # print(np.array(A).shape)
    #                  # print(A)
    #                  # print(np.array(b).shape)
    #                  n=int(room.dim[0]/self.h)
    #                  m=int(room.dim[1]/self.h)
    #                  u_new = solve_lagrangian(A,b,n,m)
    #                  # print("u_new done!", i)
    #                  # Relaxation
    #                  u_dict[i] = self.omega * u_new + (1 - self.omega) * u
    #                  # print("one room done update")
    #         self.comm.Barrier()
    #     return u_dict
    def step(self):
        print(f"Hello from process {self.rank} out of {self.size} processes")
        for k in range(self.iter):
            print("iteration : *********************",k,"rank***********",self.rank)
        #
            for i, room in enumerate(self.room_list):
                print(self.rank,"self.rank")
                # print("barrier",self.rank)
                # self.comm.Barrier()
                if self.rank == i:
                     print("Now room ", i)
                     adjacent_rooms = room.adjacent_rooms
                     # Send,Receive,Update
                     self.send_conditions(room.u, adjacent_rooms)
                     print("send complete")

                     room = self.receive_and_update(room,adjacent_rooms)

                     print("d n n complete")
                     # A, b = create_matrix_and_rhs(room, self.h)
                     # u_new = solve_lagrangian(A, b, room.n, room.m)
        #             print("neumann complete")
        #              # Relaxation
        #              room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u
                     print("one room done update")
                     self.room_list[i]=room
        return self.room_list

    def send_conditions(self, u, adjacent_rooms):
            # Possible to add top and bottom as well. Current .json layout only has left and right
            for i,adj_info in adjacent_rooms.items():
                adj_rank = adj_info['rank']
                start_pos = int(adj_info['start_pos_s']/self.h)
                end_pos = int(adj_info['end_pos_s']/self.h)
                # print(start_pos,"start pos")
                # print(end_pos, "end pos")
                direction=adj_info["direction"]
                # Send rightmost boundary if the adjacent room is to the right
                if direction == "right":
                    self.comm.send(u[start_pos:end_pos, -1], dest=adj_rank, tag=100 + self.rank)
                    # print("send to right")
                # Send the leftmost boundary if the adjacent room is to the left
                elif direction == "left":
                    self.comm.send(u[start_pos:end_pos, 0], dest=adj_rank, tag=100 + self.rank)
                    # print("send to left")
                elif direction == "top":
                    self.comm.send(u[0,start_pos:end_pos], dest=adj_rank, tag=100 + self.rank)
                    # print("send to top")
                elif direction == "bottom":
                    self.comm.send(u[-1,start_pos:end_pos], dest=adj_rank, tag=100 + self.rank)
                    # print("send to bottom")
                else:
                    raise ValueError("adj_direction is weired")


    def receive_and_update(self, room, adjacent_rooms):
            for  i,adj_info in adjacent_rooms.items():
                adj_rank = adj_info['rank']
                adj_type = adj_info['type']
                adj_direction = adj_info["direction"]
                start_pos_r=int(adj_info["start_pos_r"]/self.h)
                end_pos_r=int(adj_info["end_pos_r"]/self.h)
                start_pos_s = int(adj_info["start_pos_s"] / self.h)
                end_pos_s = int(adj_info["end_pos_s"] / self.h)
                # Receiving the boundary condition from the adjacent room
                u_received = self.comm.recv(source=adj_rank, tag=100 + adj_rank)
                adj_u=room_list[adj_rank].u
                u_len=len(u_received)
                # print("刚传进来的 u_receive",u_received,"来自 room ",adj_rank)
                # Updating boundary condition based on its type (Dirichlet or Neumann)
                if adj_type.lower() == "dirichlet":
                    if adj_direction == "right":
                        room.wall_r_right[start_pos_s:end_pos_s, -1] = u_received
                        A, b = create_matrix_and_rhs(room, self.h)
                        u_d_fertig = solve_lagrangian(A, b, room.n, room.m)
                        room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u

                    elif adj_direction == "left":
                        room.wall_r_left[start_pos_s:end_pos_s, 0] =u_received
                        A, b = create_matrix_and_rhs(room, self.h)
                        u_d_fertig = solve_lagrangian(A, b, room.n, room.m)
                        room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u

                    elif adj_direction == "top":
                        room.wall_r_top[0, start_pos_s:end_pos_s] = u_received
                        A, b = create_matrix_and_rhs(room, self.h)
                        u_d_fertig = solve_lagrangian(A, b, room.n, room.m)
                        room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u

                    elif adj_direction == "bottom":
                        room.wall_r_bottom[-1, start_pos_s:end_pos_s] = u_received
                        A, b = create_matrix_and_rhs(room, self.h)
                        u_d_fertig = solve_lagrangian(A, b, room.n, room.m)
                        room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u

                    else:
                        raise ValueError("adj_direction is weired")

                if adj_type.lower() == "neumann":
                        if adj_direction == "right":
                            room.wall_r_right[start_pos_s:end_pos_s, -1] = (adj_u[start_pos_r:end_pos_r, -2] - adj_u[start_pos_r:end_pos_r,
                                                                                            -1]) / self.h  # Compute Neumann condition
                            A, b = create_matrix_and_rhs(room, self.h)
                            u_d_fertig = solve_lagrangian(A, b, room.n, room.m)
                            room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u


                        elif adj_direction == "left":
                            room.wall_r_left[start_pos_s:end_pos_s, 0] = (adj_u[start_pos_r:end_pos_r, 1] - adj_u[start_pos_r:end_pos_r,
                                                                                         0]) / self.h  # Compute Neumann condition
                            A, b = create_matrix_and_rhs(room, self.h)
                            u_d_fertig = solve_lagrangian(A, b, room.n, room.m)
                            room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u

                        elif adj_direction == "top":
                            room.wall_r_top[0, start_pos_s:end_pos_s] = (adj_u[1, start_pos_r:end_pos_r] - adj_u[0,
                                                                                        start_pos_r:end_pos_r]) / self.h  # Compute Neumann condition
                            A, b = create_matrix_and_rhs(room, self.h)
                            u_d_fertig = solve_lagrangian(A, b, room.n, room.m)
                            room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u

                        elif adj_direction == "bottom":
                            room.wall_r_bottom[-1, start_pos_s:end_pos_s] = (adj_u[-2, start_pos_r:end_pos_r] - adj_u[-1,
                                                                                             start_pos_r:end_pos_r]) / self.h  # Compute Neumann condition
                            A, b = create_matrix_and_rhs(room, self.h)
                            u_d_fertig = solve_lagrangian(A, b, room.n, room.m)
                            room.u = self.omega * u_d_fertig + (1 - self.omega) * room.u

                        else:
                            raise ValueError("adj_direction weired")
            return room


    def visualize(self):

            av1 = self.room_list[0].u.sum() / (self.n ** 2)
            av2 = self.room_list[1].u.sum() / (2 * self.n ** 2)
            av3 = self.room_list[2].u.sum() / (self.n ** 2)
            if self.rooms == 4:
                av4 = self.room_list[3].u.sum() / (0.25 * self.n ** 2)
                av_ap = (av1 + 2 * av2 + av3 + av4 * 0.25) / 4.25
            else:
                av_ap = (av1 + 2 * av2 + av3) / 4
            print('Average room 1:', av1)
            print('Average room 2:', av2)
            print('Average room 3:', av3)
            if self.rooms == 4:
                print('Average room 4:', av4)
            print('Average appartment:', av_ap)
            U = np.ones((2 * self.n, 3 * self.n)) * (self.room_list[1].u.min() + self.room_list[1].u.max()) / 2
            U[self.n:, :self.n] = self.room_list[0].u
            U[:, self.n:2 * self.n] = self.room_list[1].u
            U[:self.n, 2 * self.n:] = self.room_list[2].u
            if self.rooms == 4:
                U[self.n:int(self.n * 1.5), self.n * 2:int(self.n * 2.5)] = self.room_list[3].u
            plt.imshow(U, cmap="twilight_shifted")
            Rec1 = Rectangle((-0.5, -0.5), self.n, self.n, color='white')
            ax = plt.gca()
            ax.add_patch(Rec1)
            if self.rooms == 4:
                Rec3 = Rectangle((self.n * 2 - 0.5, self.n * 1.5 - 0.5), self.n * 0.5, self.n * 0.5, color='white')
                Rec4 = Rectangle((self.n * 2.5 - 0.5, self.n * 1 - 0.5), self.n * 0.5, self.n, color='white')
                ax.add_patch(Rec3)
                ax.add_patch(Rec4)
            else:
                Rec2 = Rectangle((self.n * 2 - 0.5, self.n - 0.5), self.n, self.n, color='white')
                ax.add_patch(Rec2)

            plt.hlines(self.n - 0.5, -0.5, self.n - 0.5, color='black')
            plt.vlines(self.n - 0.5, -0.5, self.n * 1 - 0.5, color='black')
            plt.vlines(self.n - 0.5, self.n - 0.5, self.n * 2 - 0.5, linestyles='--', color='black')
            if self.rooms == 4:
                plt.vlines(self.n * 2.5 - 0.5, self.n - 0.5, self.n * 1.5 - 0.5, color='black')
                plt.hlines(self.n * 1.5 - 0.5, self.n * 2 - 0.5, self.n * 2.5 - 0.5, color='black')
                plt.vlines(self.n * 2 - 0.5, -0.5, self.n * 1.5 - 0.5, linestyles='--', color='black')
                plt.vlines(self.n * 2 - 0.5, self.n * 1.5 - 0.8, self.n * 2 - 0.5, color='black')
                plt.hlines(self.n - 0.5, self.n * 2.5 - 0.5, self.n * 3 - 0.5, color='black')
                plt.hlines(self.n - 0.5, self.n * 2 - 0.5, self.n * 2.5, linestyles='--', color='black')
            else:
                plt.vlines(self.n * 2 - 0.5, -0.5, self.n * 1 - 0.5, linestyles='--', color='black')
                plt.vlines(self.n * 2 - 0.5, self.n - 0.5, self.n * 2 - 0.5, color='black')
                plt.hlines(self.n - 0.5, self.n * 2 - 0.5, self.n * 3 - 0.5, color='black')

            plt.colorbar()
            plt.show()




# Navigate to dir containing MPI_extended.py and run:
#   "cd D:\MLSC\NUMN21_Project3\Project_3"
#      mpiexec -n 4 python MPI_extended.py

# set MSMPI_BIN=%CONDA_PREFIX%\Library\bin
# set MSMPI_INC=%CONDA_PREFIX%\Library\include
# set MSMPI_LIB64=%CONDA_PREFIX%\Library\lib
# mpiexec -n 4 python -m mpi4py.bench helloworld
    # Initialise MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size=comm.Get_size()
room_list=get_room_list()
h=1/20
initial_temp=20
solver =DN_solver(h, initial_temp,room_list,comm,rank,size)

room_list=solver.step()
solver.visualize()
MPI.Finalize()
print("******u_dict:",solver.rank,"\n",room_list[solver.rank].u)
# print("**************************room list[3].u after",room_list[3].u)

end_time = time.time()





