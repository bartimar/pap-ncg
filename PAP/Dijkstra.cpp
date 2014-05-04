#include "PAP.h"

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


__device__ void updateMinimumDistance(int mainIndex, bool* connected, int** vertices, int* minimumDistance, int NUMBERofVERTICES){
	for(int i=0;i< NUMBERofVERTICES; i++){
		if(connected[i] || vertices[mainIndex][i] == inf) continue;
		if(minimumDistance[mainIndex] + vertices[mainIndex][i] < minimumDistance[i]){
					minimumDistance[i] = minimumDistance[mainIndex] + vertices[mainIndex][i];
		}
	}
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
