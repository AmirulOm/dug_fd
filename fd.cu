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

__global__ void fd_kernel(double *a, double *b)
{
    size_t ind =blockIdx.y * gridDim.x * blockDim.x +  blockIdx.x * blockDim.x + threadIdx.x;
    // if(threadIdx.x == 1 )
    //     printf("%u, %u, %u, %u, %u\n",ind, threadIdx.x, blockIdx.x, blockDim.x, blockIdx.y);
    b[ind] = a[ind] + 10;
}

void fd()
{
    std::cout << "allocate\n" << std::endl;
    
    double *a_host = new double[SIZE*SIZE];
    double *a_dev, *b_dev;
    
    memset(a_host,0,sizeof(double)*SIZE*SIZE);

    size_t ind(0);

    
    for(; ind < SIZE; ++ind) 
    {
            a_host[ind] = 1.0; // Top boundary
            a_host[ind*SIZE] = 3.0; // Top boundary
            a_host[ind*SIZE + SIZE - 1] = 2.0; // Top boundary
            a_host[SIZE * (SIZE-1) + ind] = 4.0; // Top boundary
    }

// #if defined(DEBUG) 
// ind=SIZE * (SIZE-1);
// std::cout << ind << std::endl;
// for(;ind< (SIZE * (SIZE-1) + 100);++ind)
//     std::cout << a_host[ind] << " ";
// std::cout << "\n";
// #endif

    // Unnecessary copy fron a to b
    // b=a

    // Allocate device memory
    checkCuda( cudaMalloc( (void**)&a_dev, sizeof(double) * SIZE * SIZE ));
    checkCuda( cudaMalloc( (void**)&b_dev, sizeof(double) * SIZE * SIZE ));

    // Transfer device memory
    checkCuda( cudaMemcpy(a_dev, a_host, sizeof(double) * SIZE * SIZE, cudaMemcpyHostToDevice) );  

    dim3 grid(10000,10), block(1000);

    fd_kernel<<<grid,block>>>(a_dev,b_dev);
    checkCuda( cudaPeekAtLastError() );

    checkCuda( cudaMemcpy(a_host, b_dev, sizeof(double) * SIZE * SIZE, cudaMemcpyDeviceToHost) );  


    std::cout << a_host[4 + (4*SIZE)] << " " << 
                 a_host[999 + (999*SIZE)] << " " <<
                 a_host[9994 + (9994*SIZE)] << "\n";
}

int main()
{
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );
    printf("\nDevice Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    fd();

    return 0;
}