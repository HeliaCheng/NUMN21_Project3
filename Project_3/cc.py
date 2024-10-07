from mpi4py import MPI
#acts like COMM_WORLD but is a separate instance
comm = MPI.Comm.Clone( MPI.COMM_WORLD )
# print rank (the process number) and overall number of processes
print("Hello World: process", comm.Get_rank (), " out of \
",comm.Get_size ()," is reporting for duty!")