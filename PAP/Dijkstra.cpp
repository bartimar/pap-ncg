#include "PAP.h"

__global__  void dijkstra_GPU(int* vertices, int tid, int NUMBERofVERTICES, int* minimumDistance) {

	tid= threadIdx.x;
	dijkstraDistance_GPU(vertices,tid,NUMBERofVERTICES,&minimumDistance[ threadIdx.x*NUMBERofVERTICES]);	

}


__device__ void updateMinimumDistance_GPU(int mainIndex, bool* connected, int* vertices, int* minimumDistance, int NUMBERofVERTICES){
	for(int i=0;i< NUMBERofVERTICES; i++){
		if(connected[i] || vertices[mainIndex*NUMBERofVERTICES + i] == inf) continue;
		if(minimumDistance[mainIndex] + vertices[mainIndex*NUMBERofVERTICES + i] < minimumDistance[i]){
					minimumDistance[i] = minimumDistance[mainIndex] + vertices[mainIndex*NUMBERofVERTICES + i];
		}
	}
}

__device__ void findNearest_GPU(int* minimumDistance, bool *connected, int& distance, int& index,int shuf, int NUMBERofVERTICES){
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



__device__  int *dijkstraDistance_GPU(int* vertices,int shuf, int NUMBERofVERTICES,int* minimumDistance){
	
	int  distance, index;
	//printf("Shuf is %d\n",shuf);
	// start out with only node shuf connected to the tree
	bool * connected = (bool*) malloc(NUMBERofVERTICES * sizeof(bool));
	//int * minimumDistance = (int*) malloc(NUMBERofVERTICES * sizeof(int));
	
	for(int i=0; i<NUMBERofVERTICES; i++)
			connected[i] = false;
	
	connected[shuf] = true;
	
//	if(shuf==9) {
//	printf("%d: ",shuf);
//	for (int i = 0; i < NUMBERofVERTICES*NUMBERofVERTICES; i++)
//	{
//printf("%d ",connected[i-shuf*NUMBERofVERTICES]);
//	}
//	printf("\n");
//	}
//
	/*for(int i=0; i<NUMBERofVERTICES; i++)
		minimumDistance[i] = vertices[shuf*NUMBERofVERTICES+i];
*/
	for(int step=1; step<NUMBERofVERTICES; step++){
		
		findNearest_GPU(minimumDistance, connected, distance, index,shuf, NUMBERofVERTICES);

		if(distance < inf){
			connected[index] = true;
			updateMinimumDistance_GPU(index, connected, vertices, minimumDistance,NUMBERofVERTICES);
		}
	}
	
	
	
	return minimumDistance;
}



int *dijkstraDistance(int** vertices,int shuf,int NUMBERofVERTICES){
	bool *connected;
	int *minimumDistance;
	int  distance, index;

	// start out with only node shuf connected to the tree
	connected = new bool[NUMBERofVERTICES];

	for(int i=0; i<NUMBERofVERTICES; i++)
		connected[i] = false;

	connected[shuf] = true;

	// initialize the minimum distance to the one-step distance
	minimumDistance = new int[NUMBERofVERTICES];

	for(int i=0; i<NUMBERofVERTICES; i++)
		minimumDistance[i] = vertices[shuf][i];


	for(int step=1; step<NUMBERofVERTICES; step++){
		
		findNearest(minimumDistance, connected, distance, index,shuf,NUMBERofVERTICES);

		if(distance < inf){
			connected[index] = true;
			updateMinimumDistance(index, connected, vertices, minimumDistance,NUMBERofVERTICES);
		}
	}

	delete [] connected;	// free memory

	return minimumDistance;
}

void findNearest(int* minimumDistance, bool *connected, int& distance, int& index,int shuf, int NUMBERofVERTICES){
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


void updateMinimumDistance(int mainIndex, bool* connected, int** vertices, int* minimumDistance, int NUMBERofVERTICES){
	for(int i=0;i< NUMBERofVERTICES; i++){
		if(connected[i] || vertices[mainIndex][i] == inf) continue;
		if(minimumDistance[mainIndex] + vertices[mainIndex][i] < minimumDistance[i]){
					minimumDistance[i] = minimumDistance[mainIndex] + vertices[mainIndex][i];
		}
	}
}


void dijkstra(int** vertices, int** toPrint, int NUMBERofVERTICES) {
	int *minimumDistance,i;

	for(i = 0; i < NUMBERofVERTICES; i++){

		minimumDistance = dijkstraDistance(vertices,i,NUMBERofVERTICES);

		for (int j=0; j<NUMBERofVERTICES; j++){
			toPrint[i][j]=minimumDistance[j];
		}
		delete [] minimumDistance;
	}

}