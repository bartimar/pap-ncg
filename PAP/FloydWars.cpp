#include "PAP.h"

extern int NUMBERofVERTICES;

__global__ void floydWarshall_GPU_kernel(int k, int *Graph,int NUMBERofVERTICES){
	int col=blockIdx.x*blockDim.x + threadIdx.x;
	if(col>=NUMBERofVERTICES)return;
	int idx=NUMBERofVERTICES*blockIdx.y+col;

	__shared__ int best;
	if(threadIdx.x==0)
		best=Graph[NUMBERofVERTICES*blockIdx.y+k];

	__syncthreads();
	if(best==inf)return;

	int tmp_b=Graph[k*NUMBERofVERTICES+col];
	if(tmp_b==inf)return;

	int cur=best+tmp_b;
	if(cur<Graph[idx]){
		Graph[idx]=cur;
	}
}


void floydWarshall(int** vertices,int num_threads){

	//double start,end;
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

void floydWarshall_GPU(int *HostGraph, const int NUMBERofVERTICES){
	int *DeviceGraph;
	int numBytes=NUMBERofVERTICES*NUMBERofVERTICES*sizeof(int);
	cudaError_t err;

	HANDLE_ERROR ( cudaMalloc((int **)&DeviceGraph,numBytes));
	
	//copy from host (CPU) to device (GPU)
	HANDLE_ERROR (cudaMemcpy(DeviceGraph,HostGraph,numBytes,cudaMemcpyHostToDevice));
	
	dim3 dimGrid((NUMBERofVERTICES+BLOCK_SIZE-1)/BLOCK_SIZE,NUMBERofVERTICES);

	for(int k=0;k<NUMBERofVERTICES;k++){
		floydWarshall_GPU_kernel<<<dimGrid,BLOCK_SIZE>>>(k,DeviceGraph,NUMBERofVERTICES);

		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){
			printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
		}
	}

	//copy back - from device to host
	HANDLE_ERROR ( cudaMemcpy(HostGraph,DeviceGraph,numBytes,cudaMemcpyDeviceToHost));
	

	//free device memory
	HANDLE_ERROR (cudaFree(DeviceGraph));
	
}
