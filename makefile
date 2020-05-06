all:
	nvcc -std=c++11 -m64 -g -DDEBUG  fd.cu -o fd_cuda.out
	ifort fd.f90 -o fd_fort.out