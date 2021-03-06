/*
* Written by : Amirul
* email : amirul.abdullah89@gmail.com
*/

#include <cuda.h>
#include <iostream>
#include <array>
#include <assert.h>
#include <math.h> 

#ifndef SIZE
#define SIZE 10000
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1000
#endif

#define STENCIL_SIZE 3

// Cuda error checker
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) 
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void fd_kernel(double *a, double *b)
{
    // Reusable shared memory
    __shared__ double df[STENCIL_SIZE][BLOCK_SIZE];
  
    const int i = threadIdx.x;
    const int offset = blockIdx.x;
    const int globalInd = (i + (blockIdx.x * blockDim.x) - offset);
    const size_t rowOffset = SIZE; 
    int lastThread = blockDim.x - 1;
    size_t row=0;

    //Last block checker to disable thread overflow
    if( blockIdx.x == (gridDim.x -1) ){
      lastThread = rowOffset - ((blockIdx.x  * blockDim.x) - offset) - 1;
    }
    
    //copy first two row to shared memory from global
     if( globalInd < rowOffset ){
      df[0][i] = a[globalInd + (rowOffset * row)];
      df[1][i] = a[globalInd + (rowOffset * (row+1))] ;
     }
  
      for(; row < SIZE-2; ++row)
      {
        //copy third two row to shared memory from global
        if( globalInd < rowOffset ){
          df[2][i] = a[globalInd + (rowOffset * (row+2))];
        }

        __syncthreads();
  
        if( globalInd < rowOffset ){
          int id = globalInd + (rowOffset * (row+1));
          
          if(i != 0 && i != lastThread) // branch divergence on first and last thread
          {  
            b[id]= df[1][i]/2.0 + df[1][i+1]/8.0 + df[1][i-1]/8.0 +  df[0][i]/8.0 +  df[2][i]/8.0;
          }
          
          //move shared memory to load the next row.
          df[0][i] = df[1][i];
          df[1][i] = df[2][i];
        }
      }
}



void fd()
{
    double *a_host = new double[SIZE*SIZE];
    double *a_dev, *b_dev;
    
    memset(a_host,0,sizeof(double)*SIZE*SIZE);

    size_t ind(0);
    const size_t ITER_SIZE = 10;

    for(; ind < SIZE; ++ind) 
    {
            a_host[ind] = 1.0; // Top boundary
            a_host[ind*SIZE] = 3.0; // Top boundary
            a_host[ind*SIZE + SIZE - 1] = 2.0; // Top boundary
            a_host[SIZE * (SIZE-1) + ind] = 4.0; // Top boundary
    }

    // Unnecessary copy fron a to b
    // b=a

    // Allocate device memory
    checkCuda( cudaMalloc( (void**)&a_dev, sizeof(double) * SIZE * SIZE ));
    checkCuda( cudaMalloc( (void**)&b_dev, sizeof(double) * SIZE * SIZE ));

    // Transfer device memory
    checkCuda( cudaMemcpy(a_dev, a_host, sizeof(double) * SIZE * SIZE, cudaMemcpyHostToDevice) );  
    checkCuda( cudaMemcpy(b_dev, a_host, sizeof(double) * SIZE * SIZE, cudaMemcpyHostToDevice) ); 

    dim3 grid( ceil(SIZE/(float)(BLOCK_SIZE-STENCIL_SIZE+1)) ), block(BLOCK_SIZE);
    
    for (size_t iter = 0; iter < ITER_SIZE; ++iter)
      if( iter % 2 == 0) // switcher to avoid memory copy between a_dev and b_dev
        fd_kernel<<<grid,block>>>(a_dev,b_dev);
      else
        fd_kernel<<<grid,block>>>(b_dev,a_dev);
    checkCuda( cudaPeekAtLastError() );

    if((ITER_SIZE-1) % 2 == 0)
      checkCuda( cudaMemcpy(a_host, b_dev, sizeof(double) * SIZE * SIZE, cudaMemcpyDeviceToHost) );  
    else
      checkCuda( cudaMemcpy(a_host, a_dev, sizeof(double) * SIZE * SIZE, cudaMemcpyDeviceToHost) ); 

    std::cout  <<a_host[4 + (4*SIZE)] << " " << 
                 a_host[999 + (999*SIZE)] << " " <<
                 a_host[9994 + (9994*SIZE)] << "\n";
}

int main()
{
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );
    std::cout << "\nDevice Name: \n" << prop.name;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";

    fd();

    return 0;
}