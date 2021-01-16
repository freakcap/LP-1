#include<stdio.h>
#include<cuda.h>
#define row1 10 
#define col1 10 
#define row2 10 
#define col2 10 
typedef long long int LLI;

__global__ void matproductsharedmemory(LLI *l,LLI *m, LLI *n)
{
    LLI x=blockIdx.x;
    LLI y=blockIdx.y;
    __shared__ LLI p[col1];

    LLI i;
    LLI k=threadIdx.x;

    n[col2*y+x]=0;

   p[k]=l[col1*y+k]*m[col2*k+x];

  __syncthreads();

  for(i=0;i<col1;i++)
  n[col2*y+x]=n[col2*y+x]+p[i];
}

int main()
{
    LLI a[row1][col1];
    LLI b[row2][col2];
    LLI c[row1][col2];
    LLI *d,*e,*f;
    LLI i,j;

  for(i=0;i<row1;i++)
    {
        for(j=0;j<col1;j++)
            {
                a[i][j]= i*row1+j;
            }
    }

        for(i=0;i<row2;i++)
        {
            for(j=0;j<col2;j++)
                {
                  b[i][j]=i*row2+j;
                }
        }

   cudaMalloc((void **)&d,row1*col1*sizeof(LLI));
   cudaMalloc((void **)&e,row2*col2*sizeof(LLI));
   cudaMalloc((void **)&f,row1*col2*sizeof(LLI));

 cudaMemcpy(d,a,row1*col1*sizeof(LLI),cudaMemcpyHostToDevice);
 cudaMemcpy(e,b,row2*col2*sizeof(LLI),cudaMemcpyHostToDevice);

dim3 grid(col2,row1);

/* Here we are defining two dimensional Grid(collection of blocks) structure. Syntax is dim3 grid(no. of columns,no. of rows) */

matproductsharedmemory<<<grid,col1>>>(d,e,f);

 cudaMemcpy(c,f,row1*col2*sizeof(LLI),cudaMemcpyDeviceToHost);
/*
 printf("\n Product of two matrices:\n ");
    for(i=0;i<row1;i++)
    {
        for(j=0;j<col2;j++)
        {
              printf("%Ld\t",c[i][j]);
        }
        printf("\n");
    }
*/
    cudaFree(d);
    cudaFree(e);
    cudaFree(f);

    return 0;
}

/*
OUTPUT profile
==13287== NVPROF is profiling process 13287, command: ./a.out
==13287== Profiling application: ./a.out
==13287== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   94.72%  2.5322ms         1  2.5322ms  2.5322ms  2.5322ms  matproductsharedmemory(__int64*, __int64*, __int64*)
                    3.68%  98.338us         2  49.169us  49.025us  49.313us  [CUDA memcpy HtoD]
                    1.61%  42.913us         1  42.913us  42.913us  42.913us  [CUDA memcpy DtoH]
      API calls:   98.22%  189.54ms         3  63.178ms  5.3290us  189.52ms  cudaMalloc
                    1.43%  2.7661ms         3  922.02us  26.698us  2.6712ms  cudaMemcpy
                    0.19%  361.76us        94  3.8480us     170ns  233.68us  cuDeviceGetAttribute
                    0.08%  150.22us         3  50.073us  6.2080us  110.67us  cudaFree
                    0.05%  89.941us         1  89.941us  89.941us  89.941us  cuDeviceTotalMem
                    0.01%  27.216us         1  27.216us  27.216us  27.216us  cuDeviceGetName
                    0.01%  24.939us         1  24.939us  24.939us  24.939us  cudaLaunch
                    0.00%  2.2690us         3     756ns     186ns  1.7650us  cuDeviceGetCount
                    0.00%  1.0820us         2     541ns     239ns     843ns  cuDeviceGet
                    0.00%     955ns         3     318ns     172ns     542ns  cudaSetupArgument
                    0.00%     724ns         1     724ns     724ns     724ns  cudaConfigureCall
*/