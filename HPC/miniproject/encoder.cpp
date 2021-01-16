#include<iostream>
#include<mpi.h>
#include<malloc.h>
#include<string.h>

using namespace std;
#define N 20000000
char* int_to_char(int x){
	int l = 0;
	for(int i=x;i>0;i=i/10){
		l++;
	}
	char* res =(char*) malloc(l+1);
	int j =l-1;
	for(int i=x;i>0;i=i/10){
		int d = i%10;
		res[j]= '0' + d;
		j--;
	}
	res[l] ='\0';
	return res;
}

int char_to_int(char* arr,int start,int &end){
	int ans = 0;
	int i = start;
	while((int)arr[i]>47 && (int)arr[i]<58){
		ans = ans*10 + (int)arr[i] - 48;
		end++;
		i++;
	}
	return ans;
}

char* runlength(char* arr,int n)
{
	int count = 1;
	char* totalstring = (char*)malloc(n);
	totalstring[0]= arr[0];
  totalstring[1]='\0';
	for(int i=0;i<n;i++){
		if(arr[i]==arr[i+1])
			count++;
		else{
			strcat(totalstring,int_to_char(count));
			int l = strlen(totalstring);
			totalstring[l] = arr[i+1];
			totalstring[l+1]='\0';
			count = 1;
		}
	//	cout<<totalstring<<endl;
	}
  return totalstring;
}


int main(int argc,char* argv[]){

  char *arr;
  char *localArray;
  int num_elements;
  MPI_Init(&argc,&argv);
  int pid,num_proc,ierr;
  double start,finish;
  char* result;
  MPI_Status status;

  MPI_Comm_rank(MPI_COMM_WORLD,&pid);
  MPI_Comm_size(MPI_COMM_WORLD,&num_proc);



  if(pid==0){
    cout<<"no of process "<<num_proc<<endl;
    arr = new char[N];
    for(int i=0;i<N;i++){
      cin>>arr[i];
    }


		arr[N]='\0';


    num_elements = N/num_proc;
		localArray = new char[num_elements+1];
		localArray[num_elements] ='\0';
		start = MPI_Wtime();
    for(int i =1;i<num_proc;i++){
    ierr=  MPI_Send(&num_elements,1,MPI_INT,i,0,MPI_COMM_WORLD);
    }

  }
  else{

    ierr = MPI_Recv(&num_elements,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
    localArray = new char[num_elements+1];
		localArray[num_elements] ='\0';
  }

  ierr = MPI_Scatter(arr,num_elements, MPI_CHAR, localArray,num_elements, MPI_CHAR,0, MPI_COMM_WORLD);


  result=runlength(localArray,num_elements);

//	cout<<pid<<" "<<result<<endl;


const int root = 0;
int *recvcounts = NULL;
int mylen = strlen(result);

if (pid == root)
		recvcounts =(int*) malloc( num_proc * sizeof(int)) ;

MPI_Gather(&mylen, 1, MPI_INT,
					 recvcounts, 1, MPI_INT,
					 root, MPI_COMM_WORLD);

int totlen = 0;
int *displs = NULL;
char *totalstring = NULL;

if (pid == root) {
		displs = (int*)malloc( num_proc * sizeof(int) );

		displs[0] = 0;
		totlen += recvcounts[0]+1;// + 1 for '\0'

		for (int i=1; i<num_proc; i++) {
			 totlen += recvcounts[i];
			 displs[i] = displs[i-1] + recvcounts[i-1];
		}

		totalstring =(char*) malloc(totlen * sizeof(char));
		for (int i=0; i<totlen-1; i++)
				totalstring[i] = ' ';
		totalstring[totlen-1] = '\0';
}

MPI_Gatherv(result, mylen, MPI_CHAR,
						totalstring, recvcounts, displs, MPI_CHAR,
						root, MPI_COMM_WORLD);


if (pid == root) {
	//	cout<<totalstring;

		char* res = (char*)malloc(totlen-1);
		//int count = (int)totalstring[1] - 48;
		int end = 0;
		int count = char_to_int(totalstring,1,end);
		int prev = 0;
		int i = end + 1;
		res[0]=totalstring[0];
		res[1]='\0';
		while(i<totlen){
			if(totalstring[prev]==totalstring[i]){
				//count = count + (int)totalstring[i+3] - 48;
				end = 0;
				count = count + char_to_int(totalstring,i+1,end);
				prev = i;
				i=end+1+i;

			//	cout<<count<<endl;
			}else{
				strcat(res,int_to_char(count));
				int l = strlen(res);
				end = 0;
				count = char_to_int(totalstring,i+1,end);
				prev = i;
				i = end+1+i;
				res[l]=totalstring[prev];
				res[l+1]='\0';
			}

		}
		cout<<"parallel result "<<res<<endl;
		finish = MPI_Wtime();
		double pt = (finish-start);
		cout<<"parallel time: "<<pt*1000<<"ms"<<endl;
	//	free(totalstring);
		//free(displs);
	//	free(recvcounts);
	//	free(localArray);

		start = MPI_Wtime();
		//serial calulation
		result = runlength(arr,N);
		cout<<"serial result "<<result<<endl;

		finish = MPI_Wtime();
		double st = (finish-start);
		cout<<"serial time: "<<st*1000<<"ms"<<endl;
		cout<<"speedup: "<<st/pt<<endl;
		cout<<"efficiency: "<<(st/pt)/num_proc<<endl;
}
  MPI_Finalize();
  return 0;
}
