#include<stdio.h>
#include <stdlib.h>
#include<omp.h>

void merge(int array[],int low,int mid,int high)
{
	int temp[1000000];
	int i,j,k,m; 
	j=low;
	m=mid+1;
	for(i=low; j<=mid && m<=high ; i++)
	{
		if(array[j]<=array[m])
		{
			temp[i]=array[j];
			j++;
		}
		else
		{
			temp[i]=array[m];
			m++;
		}
	}
	if(j>mid)
	{
		for(k=m; k<=high; k++)
		{
			temp[i]=array[k];
			i++;
		}
	}
	else
	{
		for(k=j; k<=mid; k++)
		{
			temp[i]=array[k];
			i++;
		}
	}
	for(k=low; k<=high; k++)
		array[k]=temp[k];
}


void mergesort_Parallel(int array[],int low,int high)
{
	int mid;
	if(low<high)
	{
		mid=(low+high)/2;

   #pragma omp parallel sections num_threads(2) 
		{
      #pragma omp section
			{
				mergesort_Parallel(array,low,mid);
			}

      #pragma omp section
			{
				mergesort_Parallel(array,mid+1,high);
			}
		}
		merge(array,low,mid,high);
	}
}

void mergesort_Sequential(int array[],int low,int high)
{
	int mid;
	if(low<high)
	{
		mid=(low+high)/2;
		mergesort_Sequential(array,low,mid);
		mergesort_Sequential(array,mid+1,high);
		merge(array,low,mid,high);
	}
}


int main()
{
	int i,size;
	printf("Enter total no. of elements:\n");
	scanf("%d",&size);
	int A[size],B[size];
	for(i=0; i<size; i++)
	{
		A[i]=rand()%size;
		B[i]=A[i];
	}
	double start,end,par,seq;

	start = omp_get_wtime();
	mergesort_Parallel(A,0,size-1);
	end = omp_get_wtime();
	par = end-start;

	start = omp_get_wtime();
	mergesort_Sequential(B,0,size-1);
	end = omp_get_wtime();
	seq = end-start;

	// printf("Sorted Elements as follows:\n");
	// for(i=0; i<size; i++)
	// 	printf("%d ",A[i]);
	printf("Size - %d",size);
	printf("\n");
	printf("\n-----------------------\n Parallel Exec Time = %f",par);
	printf("\n-----------------------\n Sequential Exec Time = %f",seq);
	printf("\n\n");
	return 0;
}	