#include <cuda.h>
#include <iostream>
#include <array>
#include <assert.h>

#ifndef SIZE
#define SIZE 10000
#endif

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

// last block to handle left x-direction stencil computation
__device__ void fd_leftover_kernel(double *a, double *b,const int& row,const int& i,const int& offset,const int& rowOffset)
{
  __shared__ double dfl[3][9];

  if(threadIdx.x < blockIdx.x)
  {
    dfl[0][i] = a[i + (blockIdx.x * blockDim.x) + (rowOffset * row) - offset +  blockDim.x];
    dfl[1][i] = a[i + (blockIdx.x * blockDim.x) + (rowOffset * (row+1)) - offset + blockDim.x];
    dfl[2][i] = a[i + (blockIdx.x * blockDim.x) + (rowOffset * (row+2)) - offset + blockDim.x];
  }

  __syncthreads();

  if(threadIdx.x < blockIdx.x)
  {
    int id = i + (blockIdx.x * blockDim.x) + (rowOffset * (row+1)) - offset + blockDim.x;

    if(i != 0 && i != (9 - 1)) // branch divergence on first and last thread
    { 
      b[id]= dfl[1][i]/2.0 + dfl[1][i+1]/8.0 + dfl[1][i-1]/8.0 +  dfl[0][i]/8.0 +  dfl[2][i]/8.0;
    }
  }
}

__global__ void fd_kernel(double *a, double *b)
{
    __shared__ double df[3][1000];
  
    const int i = threadIdx.x;
    const int offset = blockIdx.x;
    const size_t rowOffset = SIZE; 
    size_t row=0;

    df[0][i] = a[i + (blockIdx.x * blockDim.x) + (rowOffset * row) - offset];
    df[1][i] = a[i + (blockIdx.x * blockDim.x) + (rowOffset * (row+1)) - offset];
   

    for(; row < SIZE-2; ++row)
    {
      df[2][i] = a[i + (blockIdx.x * blockDim.x) + (rowOffset * (row+2)) - offset];
      __syncthreads();

      int id = i + (blockIdx.x * blockDim.x) +  (rowOffset * (row+1) - offset);
      
      if(i != 0 && i != (blockDim.x - 1)) // branch divergence on first and last thread
      {  
        b[id]= df[1][i]/2.0 + df[1][i+1]/8.0 + df[1][i-1]/8.0 +  df[0][i]/8.0 +  df[2][i]/8.0;
      }

      if( blockIdx.x  == (gridDim.x - 1))
        fd_leftover_kernel(a, b, row,i, offset, rowOffset);
      
        
      df[0][i] = df[1][i];
      df[1][i] = df[2][i];
      
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

    dim3 grid(10), block(1000);
    
    for (size_t iter = 0; iter < ITER_SIZE; ++iter)
      if( iter % 2 == 0)
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
    std::cout << "maxThreadsDim    : " << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] << "\n";
    std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n";

    fd();

    return 0;
}