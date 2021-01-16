#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define lli long long

void swap(lli *num1, lli *num2)
{

	lli temp = *num1;
	*num1 =  *num2;
	*num2 = temp;
}

int main () {

	lli SZ = 300000;
	lli A[SZ],B[SZ];
	for(lli i=0;i<SZ;i++)
	{
		A[i]=rand()%SZ;
		B[i]=A[i];
	}
	//lli A[5] = {6,9,1,3,7};
	lli N = SZ;
	lli i=0, j=0; 
	lli temp;
	lli first;
	double start,end;
	start = omp_get_wtime();
	#pragma omp parallel num_threads(4) default(none) shared(A,N) private(i,temp,j)
		{
			for (i = 0; i < N; i++)
			{
				//even phase
				if (i % 2 == 0)
				{
					#pragma omp for
					for (j = 1; j < N; j += 2)
					{
						if (A[j - 1] > A[j])
						{
							temp = A[j];
							A[j] = A[j - 1];
							A[j - 1] = temp;
						}
					}
				}
				//odd phase
				else
				{
					#pragma omp for
					for (j = 1; j < N - 1; j += 2)
					{
						if (A[j] > A[j + 1])
						{
							temp = A[j];
							A[j] = A[j + 1];
							A[j + 1] = temp;
						}
					}
				}
			}
		}
	end = omp_get_wtime();
	// for(i=0;i<N;i++)
	// {
	// 	printf(" %lld",A[i]);
	// }
	printf("Size -  %lld",N);
	printf("\n-----------------------\n Parallel Exec Time = %f",(end-start));

	start = omp_get_wtime();
	for( i = 0; i < N-1; i++ )
	{
		for( j = 0; j < N-1; j++ )
		{
			if( B[j] > B[j+1] )
			{
				swap( &B[j], &B[j+1] );
			}
		}
	}
	end = omp_get_wtime();
	printf("\n-----------------------\n Sequential Exec Time = %f",(end-start));
}