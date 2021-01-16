#include<iostream>
#include<cstdlib>
#include<cmath>
#include<time.h>
using namespace std;


__global__ void matrixVectorMultiplication(int *a, int *b, int *c, int n)
{
    int row=threadIdx.x+blockDim.x*blockIdx.x;
    int sum=0;
   
    if(row<n){
        for(int j=0;j<n;j++)
        {
            sum=sum+a[(j*n)+row]*b[j];
        }
    c[row]=sum;
    }
}
int main()
{
    int *a,*b,*c;
    int *a_dev,*b_dev,*c_dev;
    int n=10;
    
    a=new int[n*n];
    b=new int[n];
    c=new int[n];
    int *d=new int[n];
    int size=n*sizeof(int);
    cudaMalloc(&a_dev,size*size);
    cudaMalloc(&b_dev,size);
    cudaMalloc(&c_dev,size);
    

    cout<<"\n\nMatrix is :\n\n";
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            a[i*n+j]= i*n+j+1; //rand()%n;
            cout<<a[i*n+j]<<" ";
        }
        
        b[i]= i+1; //rand()%n;
        cout<<"\n";
       // d[i]=a[i]+b[i];
    }
    
    cout<<"\n\nVector is: \n\n";
    for(int i=0;i<n;i++)
        cout<<b[i]<<" ";
    cout<<"\n\n";
    cudaMemcpy(a_dev,a,size*size,cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev,b,size,cudaMemcpyHostToDevice);
    
    
    dim3 threadsPerBlock(n, n);
    dim3 blocksPerGrid(1, 1);
    
    if(n*n>512){
        threadsPerBlock.x=512;
        threadsPerBlock.y=512;
        blocksPerGrid.x=ceil((double)n/(double)threadsPerBlock.x);
        blocksPerGrid.y=ceil((double)n/(double)threadsPerBlock.y);
    }
    
    matrixVectorMultiplication<<<n/256 +1,256>>>(a_dev,b_dev,c_dev,n);
    
    cudaMemcpy(c,c_dev,size,cudaMemcpyDeviceToHost);
    
    //CPU matrixVector multiplication
    clock_t t=clock();
    int sum=0;
    for(int row=0;row<n;row++)
    {
        sum=0;
        for(int col=0;col<n;col++)
        {
              sum=sum+a[col*n+row]*b[col];  
            
        }
      d[row]=sum;
    }
    t=clock()-t;
        cout<<"\nCPU Time Elapsed:  "<<((double)t);      //((double)t)/CLOCKS_PER_SEC;

    
    int error=0;
    cout<<"\n\n";
    for(int i=0;i<n;i++){
        error+=d[i]-c[i];
       cout<<" gpu "<<c[i]<<" CPU "<<d[i]<<endl;
    }
    
    cout<<"\nError : "<<error<<"\n\n";
    
    
    return 0;
}

/*
Output
==11960== NVPROF is profiling process 11960, command: ./a.out


Matrix is :

1 2 3 4 5 6 7 8 9 10 
11 12 13 14 15 16 17 18 19 20 
21 22 23 24 25 26 27 28 29 30 
31 32 33 34 35 36 37 38 39 40 
41 42 43 44 45 46 47 48 49 50 
51 52 53 54 55 56 57 58 59 60 
61 62 63 64 65 66 67 68 69 70 
71 72 73 74 75 76 77 78 79 80 
81 82 83 84 85 86 87 88 89 90 
91 92 93 94 95 96 97 98 99 100 


Vector is: 

1 2 3 4 5 6 7 8 9 10 


CPU Time Elapsed:  3

 gpu 3355 CPU 3355
 gpu 3410 CPU 3410
 gpu 3465 CPU 3465
 gpu 3520 CPU 3520
 gpu 3575 CPU 3575
 gpu 3630 CPU 3630
 gpu 3685 CPU 3685
 gpu 3740 CPU 3740
 gpu 3795 CPU 3795
 gpu 3850 CPU 3850

Error : 0

==11960== Profiling application: ./a.out
==11960== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   48.69%  4.1910us         1  4.1910us  4.1910us  4.1910us  matrixVectorMultiplication(int*, int*, int*, int)
                   30.86%  2.6560us         2  1.3280us  1.0880us  1.5680us  [CUDA memcpy HtoD]
                   20.45%  1.7600us         1  1.7600us  1.7600us  1.7600us  [CUDA memcpy DtoH]
      API calls:   99.71%  200.37ms         3  66.791ms  5.1320us  200.36ms  cudaMalloc
                    0.16%  313.26us        97  3.2290us     124ns  206.98us  cuDeviceGetAttribute
                    0.05%  104.47us         1  104.47us  104.47us  104.47us  cuDeviceTotalMem
                    0.03%  64.413us         1  64.413us  64.413us  64.413us  cuDeviceGetName
                    0.03%  62.632us         3  20.877us  19.062us  22.537us  cudaMemcpy
                    0.02%  32.072us         1  32.072us  32.072us  32.072us  cudaLaunchKernel
                    0.00%  4.1760us         1  4.1760us  4.1760us  4.1760us  cuDeviceGetPCIBusId
                    0.00%  2.1090us         3     703ns     121ns  1.7920us  cuDeviceGetCount
                    0.00%     988ns         2     494ns     137ns     851ns  cuDeviceGet
                    0.00%     210ns         1     210ns     210ns     210ns  cuDeviceGetUuid

*/