#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <omp.h>
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdio.h>

using namespace std;

#define BLOCK_SIZE 256

int NUMBERofVERTICES;
const int inf = INT_MAX;
__device__ int *dijkstraDistance (int** vertices, int NUMBERofVERTICES);
__device__ void findNearest (int* minimumDistance, bool* connected, int& d, int& v,int, int NUMBERofVERTICES);
void init (int**& vertices,int=0, int=1);
__device__ void updateMinimumDistance (int mv, bool* connected, int** vertices, int* minimumDistance, int NUMBERofVERTICES);


static void HandleError( cudaError_t err, const char *file, int line ) {
 if (err != cudaSuccess) { 
 printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line ); 
 exit( EXIT_FAILURE ); 
}} 
 
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 



void floydWarshall(int** vertices,int num_threads){

	double start,end;
	//start=omp_get_wtime();
	for(int k=0; k<NUMBERofVERTICES; k++) {


		//#pragma omp master
		//omp_set_dynamic(0);     // Explicitly disable dynamic teams
		//omp_set_num_threads(num_threads); // Use x threads for all consecutive parallel regions
		int i,j;

		//#pragma omp parallel for private(i,j), shared(k)
		for(i=0; i<NUMBERofVERTICES; i++){
			if(vertices[i][k] == inf) continue;
			for (j=0; j<NUMBERofVERTICES; j++){
				if(vertices[k][j] == inf || i == j) continue;
				if(vertices[i][k] + vertices[k][j] < vertices[i][j]){
					vertices[i][j] = vertices[i][k] + vertices[k][j];
				}
			}
		}
	}

	//end=omp_get_wtime();


	//cout<< "Time CPU_Warshall: "<< end-start <<endl;
}

__global__ void _Wake_GPU(int reps){
	int idx=blockIdx.x*blockDim.x + threadIdx.x;
	if(idx>=reps) return;
}

__global__ void floydWarshall_GPU_kernel(int k, int *G,int N){
	int col=blockIdx.x*blockDim.x + threadIdx.x;
	if(col>=N)return;
	int idx=N*blockIdx.y+col;

	__shared__ int best;
	if(threadIdx.x==0)
		best=G[N*blockIdx.y+k];

	__syncthreads();
	if(best==inf)return;

	int tmp_b=G[k*N+col];
	if(tmp_b==inf)return;

	int cur=best+tmp_b;
	if(cur<G[idx]){
		G[idx]=cur;
	}
}

void floydWarshall_GPU(int *HostGraph, const int NUMBERofVERTICES){
	int *DeviceGraph;
	int numBytes=NUMBERofVERTICES*NUMBERofVERTICES*sizeof(int);

	cudaError_t err=cudaMalloc((int **)&DeviceGraph,numBytes);
	if(err!=cudaSuccess){
		printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
	}

	//copy from host (CPU) to device (GPU)
	err=cudaMemcpy(DeviceGraph,HostGraph,numBytes,cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){
		printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
	}

	dim3 dimGrid((NUMBERofVERTICES+BLOCK_SIZE-1)/BLOCK_SIZE,NUMBERofVERTICES);

	for(int k=0;k<NUMBERofVERTICES;k++){
		floydWarshall_GPU_kernel<<<dimGrid,BLOCK_SIZE>>>(k,DeviceGraph,NUMBERofVERTICES);

		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){
			printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
		}
	}

	//copy back - from device to host
	err=cudaMemcpy(HostGraph,DeviceGraph,numBytes,cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){
		printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
	}

	//free device memory
	err=cudaFree(DeviceGraph);
	if(err!=cudaSuccess){
		printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
	}
}

void printVertices(int** vertices) {
	cout << "    ";
	for(int i=0; i < NUMBERofVERTICES; i++)	cout << setw(4) << char ('A' + i); 
	cout<<endl;

	for(int i=0; i < NUMBERofVERTICES; i++){
		cout << setw(4) << char ('A' + i);

		for(int j=0; j < NUMBERofVERTICES; j++){
			if(vertices[i][j] == inf) cout << setw(4) << "Inf";
			else cout << setw(4) <<  vertices[i][j];

		}
		cout<< endl;
	}
	cout<<endl;
}

void printVertices_GPU(int* D_G) {
	cout << "    ";
	for(int i=0; i < NUMBERofVERTICES; i++)	cout << setw(4) << char ('A' + i); 
	cout<<endl;

	for(int i=0; i < NUMBERofVERTICES; i++){
		cout << setw(4) << char ('A' + i);

		for(int j=0; j < NUMBERofVERTICES; j++){
			if(D_G[i*NUMBERofVERTICES+j] == inf) cout << setw(4) << "Inf";
			else cout << setw(4) << D_G[i*NUMBERofVERTICES+j];

		}
		cout<< endl;
	}
	cout<<endl;
}


__device__  int *dijkstraDistance(int** vertices,int shuf, int NUMBERofVERTICES){
	bool *connected;
	int *minimumDistance;
	int  distance, index;

	// start out with only node shuf connected to the tree
	connected= (bool*) malloc(NUMBERofVERTICES * sizeof(bool) ); 



	for(int i=0; i<NUMBERofVERTICES; i++)
		connected[i] = false;

	connected[shuf] = true;

	// initialize the minimum distance to the one-step distance
	minimumDistance= (int* )malloc(NUMBERofVERTICES * sizeof(int) ); 

	for(int i=0; i<NUMBERofVERTICES; i++)
		minimumDistance[i] = vertices[shuf][i];

	for(int step=1; step<NUMBERofVERTICES; step++){
		
		findNearest(minimumDistance, connected, distance, index,shuf, NUMBERofVERTICES);

		if(distance < inf){
			connected[index] = true;
			updateMinimumDistance(index, connected, vertices, minimumDistance,NUMBERofVERTICES);
		}
	}

	free( connected ); 

	return minimumDistance;
}

__device__ void findNearest(int* minimumDistance, bool *connected, int& distance, int& index,int shuf, int NUMBERofVERTICES){
	// output: 
	//	- int distance, the distance from node 0 to the nearest unconnected node in the range first to last
	//	- int index, the index of the nearest unconnected node in the range first to last.

	distance = inf;
	index = -1;

	for(int i=0; i<NUMBERofVERTICES; i++){
		if(i==shuf) continue;
		if(!connected[i] && minimumDistance[i] < distance){
			distance = minimumDistance[i];
			index = i;
		}
	}
}


void alloc2Darray(int**& arr, int*& arr_GPU) {

	arr=new int*[NUMBERofVERTICES];
	arr_GPU=new int[NUMBERofVERTICES*NUMBERofVERTICES];

	for (int i = 0; i < NUMBERofVERTICES; i++)
		arr[i]=new int[NUMBERofVERTICES];
}

void dealloc2Darray(int**& arr, int*& arr_GPU) {

	for (int i = 0; i < NUMBERofVERTICES; i++)
		delete arr[i];

	delete [] arr;
	delete [] arr_GPU;
	arr=NULL;
	arr_GPU=NULL;
}

void init(int**& vertices,int shuf, int example, int*& D_G){

	static bool initialized = false;

	if(!initialized) {
		for(int i=0; i<NUMBERofVERTICES; i++){
			for(int j=0; j<NUMBERofVERTICES; j++){
				if(i==j){
					vertices[i][i] = 0;
					D_G[i*NUMBERofVERTICES+j] = 0;
				}else{	// inicialization of all the other vertices to inf
					vertices[i][j] = inf;
					D_G[i*NUMBERofVERTICES+j] = inf;
				}	
			}

		}
	}
	switch(example){

	case 1:
		vertices[(0+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = vertices[(1+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 40;
		vertices[(0+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = vertices[(2+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 15;
		vertices[(1+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = vertices[(2+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 20;
		vertices[(1+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = vertices[(3+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 10;
		vertices[(1+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = vertices[(4+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 25;
		vertices[(2+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = vertices[(3+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 100;
		vertices[(1+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = vertices[(5+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 6;
		vertices[(4+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = vertices[(5+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = 8;

		D_G[1]= D_G[NUMBERofVERTICES] = 40;
		D_G[2] = D_G[2*NUMBERofVERTICES] = 15;
		D_G[NUMBERofVERTICES+2] = D_G[2*NUMBERofVERTICES+1] = 20;
		D_G[NUMBERofVERTICES+3] = D_G[3*NUMBERofVERTICES+1] = 10;
		D_G[NUMBERofVERTICES+4] = D_G[4*NUMBERofVERTICES+1] = 25;
		D_G[2*NUMBERofVERTICES+3] = D_G[3*NUMBERofVERTICES+2] = 100;
		D_G[NUMBERofVERTICES+5] = D_G[5*NUMBERofVERTICES+1] = 6;
		D_G[4*NUMBERofVERTICES+5] = D_G[5*NUMBERofVERTICES+4] = 8;
		break;
	case 2:
		vertices[(0+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 8;
		vertices[(0+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = 1;
		vertices[(0+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = 2;
		vertices[(1+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = 8;
		vertices[(2+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 8;
		vertices[(2+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 9;
		vertices[(2+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = 2;
		vertices[(2+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = 3;
		vertices[(3+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 1;
		vertices[(3+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 2;
		vertices[(4+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 2;
		vertices[(5+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 8;
		vertices[(5+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 3;
		vertices[(5+shuf)%NUMBERofVERTICES][(6+shuf)%NUMBERofVERTICES] = 2;
		break;
	case 3:
		vertices[(0+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 8;
		vertices[(0+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = 1;
		vertices[(0+shuf)%NUMBERofVERTICES][(11+shuf)%NUMBERofVERTICES] = 1;
		vertices[(0+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = 2;
		vertices[(1+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = 8;
		vertices[(2+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 8;
		vertices[(2+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 9;
		vertices[(2+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = 2;
		vertices[(2+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = 3;
		vertices[(3+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 1;
		vertices[(3+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 2;
		vertices[(4+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 2;
		vertices[(5+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 8;
		vertices[(5+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 3;
		vertices[(6+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = 22;
		vertices[(6+shuf)%NUMBERofVERTICES][(8+shuf)%NUMBERofVERTICES] = 102;
		vertices[(6+shuf)%NUMBERofVERTICES][(9+shuf)%NUMBERofVERTICES] = 8;
		vertices[(6+shuf)%NUMBERofVERTICES][(10+shuf)%NUMBERofVERTICES] = 9;
		vertices[(6+shuf)%NUMBERofVERTICES][(11+shuf)%NUMBERofVERTICES] = 11;
		vertices[(6+shuf)%NUMBERofVERTICES][(12+shuf)%NUMBERofVERTICES] = 22;
		vertices[(7+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 8;
		vertices[(7+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = 3;
		vertices[(7+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = 6;
		vertices[(7+shuf)%NUMBERofVERTICES][(8+shuf)%NUMBERofVERTICES] = 7;
		vertices[(7+shuf)%NUMBERofVERTICES][(9+shuf)%NUMBERofVERTICES] = 7;
		vertices[(7+shuf)%NUMBERofVERTICES][(11+shuf)%NUMBERofVERTICES] = 11;
		vertices[(8+shuf)%NUMBERofVERTICES][(11+shuf)%NUMBERofVERTICES] = 10;
		vertices[(8+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = 1;
		vertices[(8+shuf)%NUMBERofVERTICES][(6+shuf)%NUMBERofVERTICES] = 17;
		vertices[(8+shuf)%NUMBERofVERTICES][(10+shuf)%NUMBERofVERTICES] = 3;
		vertices[(9+shuf)%NUMBERofVERTICES][(6+shuf)%NUMBERofVERTICES] = 4;
		vertices[(10+shuf)%NUMBERofVERTICES][(6+shuf)%NUMBERofVERTICES] = 20;
		vertices[(11+shuf)%NUMBERofVERTICES][(6+shuf)%NUMBERofVERTICES] = 1;
		vertices[(12+shuf)%NUMBERofVERTICES][(6+shuf)%NUMBERofVERTICES] = 3;
		vertices[(11+shuf)%NUMBERofVERTICES][(7+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(12+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(8+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(7+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(7+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(7+shuf)%NUMBERofVERTICES] = 7;
		vertices[(11+shuf)%NUMBERofVERTICES][(7+shuf)%NUMBERofVERTICES] = 7;
		break;
	case 0: //random
		int val;
		if(initialized) break;
		for (int i = 0; i < NUMBERofVERTICES; i++)
			for (int j = 0; j < NUMBERofVERTICES; j++)
				vertices[i][j]= (val=rand()%150) ? val : inf ;

		initialized=true;
	}


}

__device__ void updateMinimumDistance(int mainIndex, bool* connected, int** vertices, int* minimumDistance, int NUMBERofVERTICES){
	for(int i=0;i< NUMBERofVERTICES; i++){
		if(connected[i] || vertices[mainIndex][i] == inf) continue;
		if(minimumDistance[mainIndex] + vertices[mainIndex][i] < minimumDistance[i]){
					minimumDistance[i] = minimumDistance[mainIndex] + vertices[mainIndex][i];
		}
	}
}
void initExample(int& example) {

	cout<<"Enter example: 1, 2, 3 or 0 (for random)"<<endl;
	cin>>example;


	switch(example) {
	case 1:
		NUMBERofVERTICES=6;
		break;
	case 2:
		NUMBERofVERTICES=7;
		break;
	case 3:
		NUMBERofVERTICES=13;
		break;
	case 0:
		cout<<"Enter number of vertices: "<<endl;
		cin>>NUMBERofVERTICES;
		if(NUMBERofVERTICES < 1)  {
			cout<< "Number of vertices must be > 0"<< endl;
			exit(1);
		}
		break;
	default:
		cout<<"Bad parameter. Exit"<<endl;
		exit(1);
	}

}

void initThreads(int& num_threads) {

	//cout<<"Enter number of threads: "<<endl;
	//cin>>num_threads;
	num_threads = 1;
}
void printInput(int** vertices) {
	cout  << endl << "Input matrix of distances" << endl;

	for(int k=0; k<NUMBERofVERTICES; k++){
		for(int j=0; j<NUMBERofVERTICES; j++){
			if(vertices[k][j] == inf) cout << setw(6) << "Inf";
			else cout << setw(6) <<  vertices[k][j];
		}
		cout << endl;
	}
}

void printInput_GPU(int* D_G) {
	cout  << endl << "Input matrix GPU of distances" << endl;

	for(int k=0; k<NUMBERofVERTICES; k++){
		for(int j=0; j<NUMBERofVERTICES; j++){
			if(D_G[k*NUMBERofVERTICES+j] == inf) cout << setw(6) << "Inf";
			else cout << setw(6) <<  D_G[k*NUMBERofVERTICES+j];
		}
		cout << endl;
	}
}
__global__  void dijkstra(int** vertices, int** toPrint, int example, int NUMBERofVERTICES) {
	

	//int i=blockIdx.x; 
	//int j=threadIdx.x; 

	int *minimumDistance,i;
	for(i = 0; i < NUMBERofVERTICES; i++){

		minimumDistance = dijkstraDistance(vertices,i,NUMBERofVERTICES);

		//for (int j=0; j<NUMBERofVERTICES; j++){
		//	//toPrint[i][j]=minimumDistance[j];
		//}
		//delete [] minimumDistance;
	}
	
}

int main(int argc, char** argv){
	//CUDA - spoustet na 1,2,4,6,8,12,24
	int **vertices=NULL,**toPrint=NULL;
	int example,num_threads;
	int i;
	srand((unsigned int)time(NULL));
	int *D_G=NULL;

	/*if(argc<2){
	cout<< "Error. Too few parameters.\nUSAGE: "<< argv[0] <<" numberOfVertices threads"<<endl;
	exit(1);
	}
	example=0;
	NUMBERofVERTICES=atoi(argv[1]);
	num_threads=atoi(argv[2]);

	cout<< "Starting computation. Number of vertices=" << NUMBERofVERTICES <<" threads=" << num_threads << endl;
	*/

	initExample(example);
	initThreads(num_threads);

	alloc2Darray(vertices, D_G);
	//alloc2Darray(toPrint);
	// inicialization of data
	init(vertices, 0, example, D_G);	

	// print input
	//printInput(vertices);

	
	//cout << endl << endl << " Dijkstra" << endl;
	//printVertices(toPrint);

	printInput(vertices);
	printInput_GPU(D_G);

	bool same=true;
	for (int i = 0; i < NUMBERofVERTICES; i++){
		for (int j = 0; j < NUMBERofVERTICES; j++){
			if(vertices[i][j] != D_G[i*NUMBERofVERTICES+j]){ 
				same=false;
				//cout<< "Error at vertices["<< i<<"]["<< j<<"]" <<endl;
				//break;
			}
		}
	}

	if(same) cout << endl << "Inputs are the same." << endl << endl;
	else cout << endl << "Inputs are not the same." << endl << endl;

	//launch FloydWarshall
	cout << " FloydWarshall_CPU" << endl;
	floydWarshall(vertices,num_threads);
	printVertices(vertices);

	cout << " FloydWarshall_GPU" << endl;
	_Wake_GPU<<<1,BLOCK_SIZE>>>(32);
	floydWarshall_GPU(D_G,NUMBERofVERTICES);
	printVertices_GPU(D_G);
	//check if the outputs were the same
	same=true;
	for (int i = 0; i < NUMBERofVERTICES; i++){
		for (int j = 0; j < NUMBERofVERTICES; j++){
			if(vertices[i][j] != D_G[i*NUMBERofVERTICES+j]){ 
				same=false;
				//cout<< "Error at vertices["<< i<<"]["<< j<<"]" <<endl;
				//break;
			}
		}
	}

	if(same) cout << "Results are the same." <<endl;
	else cout << "Results are not the same." <<endl;


	int ** devVertices;
	alloc2Darray(devVertices,D_G);

	for (int i = 0; i < NUMBERofVERTICES; i++)
	{
		cudaMemcpy(devVertices[i],vertices[i],sizeof(int)*NUMBERofVERTICES,cudaMemcpyHostToDevice); 
	}

	cudaEvent_t start, stop; 
	float elapsedTime; 
	cudaEventCreate( &start ) ; 
	cudaEventCreate( &stop ) ; 
	cudaEventRecord( start, 0 );

	//launch Dijkstra
	dijkstra<<<1,NUMBERofVERTICES>>>(devVertices,toPrint,example, NUMBERofVERTICES);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop ) ; 
	cudaEventElapsedTime( &elapsedTime, start, stop ); 
	cout << "GPU time taken: "<< elapsedTime <<  " ms" << endl; 
	

	cudaThreadSynchronize(); 





	//if(same) cout <<"NoV="<< NUMBERofVERTICES<<": Dijkstra and FloydWarshall outputs are the same. OK!" << endl << endl;
	//else cout << "NoV="<< NUMBERofVERTICES<<": Dijkstra and FloydWarshall outputs are not the same. ERROR!" << endl << endl;


	dealloc2Darray(vertices, D_G);
	//	dealloc2Darray(toPrint);

	//system ("pause");
	return 0;
}