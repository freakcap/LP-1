#include<iostream>
#include<cstdlib>
#include<cmath>
#include<time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

using namespace std;

__global__ void vector_add(float *out, float *a, float *b, int n) {
   
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   if(i<N){
    out[i]=a[i]+b[i];
   }
}

int main(){
    float *a, *b, *out,*cpu_out;
    float *d_a, *d_b, *d_out; 
    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    cpu_out = (float*)malloc(sizeof(float) * N);
    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = i*1.0f;
        b[i] = i*1.0f;
    }

    // Allocate device memory 
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel 

    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    clock_t t=clock();
    for(int i=0;i<N;i++){
        cpu_out[i] = a[i]+b[i];
    }
     t=clock()-t;
        cout<<"\nCPU Time Elapsed:  "<<((double)t)<<"\n"; 

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED\n");

	// for(int i=0;i<N;i++)
	// printf("%lf ",out[i]);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}

/*
CPU Time Elapsed:  41444
PASSED
==12102== Profiling application: ./a.out
==12102== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.13%  48.575ms         2  24.287ms  24.072ms  24.502ms  [CUDA memcpy HtoD]
                   29.47%  23.809ms         1  23.809ms  23.809ms  23.809ms  [CUDA memcpy DtoH]
                   10.39%  8.3949ms         1  8.3949ms  8.3949ms  8.3949ms  vector_add(float*, float*, float*, int)
      API calls:   68.41%  207.09ms         3  69.028ms  161.39us  206.76ms  cudaMalloc
                   27.03%  81.812ms         3  27.271ms  24.179ms  33.384ms  cudaMemcpy
                    4.22%  12.782ms         3  4.2606ms  187.75us  8.4342ms  cudaFree
                    0.24%  739.10us        97  7.6190us     124ns  328.08us  cuDeviceGetAttribute
                    0.05%  155.08us         1  155.08us  155.08us  155.08us  cuDeviceTotalMem
                    0.03%  80.206us         1  80.206us  80.206us  80.206us  cuDeviceGetName
                    0.01%  29.702us         1  29.702us  29.702us  29.702us  cudaLaunchKernel
                    0.00%  3.8710us         1  3.8710us  3.8710us  3.8710us  cuDeviceGetPCIBusId
                    0.00%  2.0160us         3     672ns     140ns  1.3220us  cuDeviceGetCount
                    0.00%  1.0650us         2     532ns     196ns     869ns  cuDeviceGet
                    0.00%     206ns         1     206ns     206ns     206ns  cuDeviceGetUuid

*/