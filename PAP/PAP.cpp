#include "stdafx.h"


# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <ctime>
# include <omp.h>

using namespace std;

# define NUMBERofVERTICES 7
const int inf = INT_MAX;

int *dijkstraDistance (int vertices[NUMBERofVERTICES][NUMBERofVERTICES]);
void findNearest (int s, int e, int minimumDistance[NUMBERofVERTICES], bool connected[NUMBERofVERTICES], int *d, int *v);
void init (int vertices[NUMBERofVERTICES][NUMBERofVERTICES],int=0, int=1);
void updateMinimumDistance (int s, int e, int mv, bool connected[NUMBERofVERTICES], int vertices[NUMBERofVERTICES][NUMBERofVERTICES], int minimumDistance[NUMBERofVERTICES]);

void floydWarshall(int vertices[NUMBERofVERTICES][NUMBERofVERTICES]){
	int i, j, k;
	for(k=0; k<NUMBERofVERTICES; k++) {
		for(i=0; i<NUMBERofVERTICES; i++){
			for (j=0; j<NUMBERofVERTICES; j++){
				if(vertices[i][k] == inf || vertices[k][j] == inf || i == j) continue;

				if((vertices[i][k] + vertices[k][j] < vertices[i][j]) ||(vertices[i][j] == 0)){
					vertices[i][j] = vertices[i][k] + vertices[k][j];
				}
			}
		}
	}

}

void printVertices(int vertices[NUMBERofVERTICES][NUMBERofVERTICES]) {

	cout<< "    ";
	for(int i=0; i < NUMBERofVERTICES; i++)	cout << setw(4) << char ('A' + i); 
	cout<<endl;

	for(int i=0; i < NUMBERofVERTICES; i++){
		cout << setw(4) << char ('A' + i);

		for(int j=0; j < NUMBERofVERTICES; j++){
			if(vertices[i][j] == inf){
				cout << setw(4) << "Inf";
			}else{
				cout << setw(4) <<  vertices[i][j];
			}
		}
		cout<< endl;
	}
	cout<<endl;

}

int *dijkstraDistance(int vertices[NUMBERofVERTICES][NUMBERofVERTICES]){
	bool *connected;
	int i, *minimumDistance;
	int mainDistance, mainIndex;
	int myFirst = 1, myLast=NUMBERofVERTICES-1, distance, index, myStep;

	// start out with only node 0 connected to the tree
	connected = new bool[NUMBERofVERTICES];
	connected[0] = true;

	for(i=1; i<NUMBERofVERTICES; i++){
		connected[i] = false;
	}

	// initialize the minimum distance to the one-step distance
	minimumDistance = new int[NUMBERofVERTICES];

	for(i=0; i<NUMBERofVERTICES; i++){
		minimumDistance[i] = vertices[0][i];
	}

	for(myStep=1; myStep<NUMBERofVERTICES; myStep++){
		mainDistance = inf;
		mainIndex = -1; 

		findNearest(myFirst, myLast, minimumDistance, connected, &distance, &index);

		if(distance < mainDistance){
			mainDistance = distance;
			mainIndex = index;
		}

		if(mainIndex != - 1){
			connected[mainIndex] = true;
			updateMinimumDistance(myFirst, myLast, mainIndex, connected, vertices, minimumDistance);
		}
	}

	delete [] connected;	// free memory

	return minimumDistance;
}

void findNearest(int first, int last, int minimumDistance[NUMBERofVERTICES], bool connected[NUMBERofVERTICES], int *distance, int *index){
	// output: 
	//	- int *distance, the distance from node 0 to the nearest unconnected node in the range first to last
	//	- int *index, the index of the nearest unconnected node in the range first to last.

	*distance = inf;
	*index = -1;

	for(int i=first; i<=last; i++){
		if(!connected[i] && minimumDistance[i] < *distance){
			*distance = minimumDistance[i];
			*index = i;
		}
	}
}

void init(int vertices[NUMBERofVERTICES][NUMBERofVERTICES],int shuf, int example){
	for(int i=0; i<NUMBERofVERTICES; i++){
		for(int j=0; j<NUMBERofVERTICES; j++){
			if(i==j){					// the same vertices, distance = 0			
				vertices[i][i] = 0;

			}else{
				vertices[i][j] = inf;	// inicialization of all the other vertices to inf
			}
		}
	}

	if(example == 1){
		vertices[(0+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = vertices[(1+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 40;
		vertices[(0+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = vertices[(2+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 15;
		vertices[(1+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = vertices[(2+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 20;
		vertices[(1+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = vertices[(3+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 10;
		vertices[(1+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = vertices[(4+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 25;
		vertices[(2+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = vertices[(3+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 100;
		vertices[(1+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = vertices[(5+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 6;
		vertices[(4+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = vertices[(5+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = 8;
	}else if(example == 2){
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
	}
}

void updateMinimumDistance(int first, int last, int mainIndex, bool connected[NUMBERofVERTICES], int vertices[NUMBERofVERTICES][NUMBERofVERTICES], int minimumDistance[NUMBERofVERTICES]){
	for(int i=first; i <= last; i++){
		if(!connected[i]){

			if(vertices[mainIndex][i] < inf){

				if(minimumDistance[mainIndex] + vertices[mainIndex][i] < minimumDistance[i]){
					minimumDistance[i] = minimumDistance[mainIndex] + vertices[mainIndex][i];
				}
			}
		}
	}
}

int _tmain(int argc, _TCHAR* argv[]){
	int *minimumDistance;
	int vertices[NUMBERofVERTICES][NUMBERofVERTICES];
	int toPrint[NUMBERofVERTICES][NUMBERofVERTICES];
	int example=2;
	// inicialization of data
	init(vertices, 0, example);	

	// print input
	cout << "Input matrix of distances" << endl << endl;

	for(int k=0; k<NUMBERofVERTICES; k++){
		for(int j=0; j<NUMBERofVERTICES; j++){
			if(vertices[k][j] == inf){
				cout << setw(6) << "Inf";

			}else{
				cout << setw(6) <<  vertices[k][j];
			}
		}
		cout << endl;
	}

	for(int i = 0; i < NUMBERofVERTICES; i++){
		// inicialization of data
		init(vertices, -i+NUMBERofVERTICES, example);	

		minimumDistance = dijkstraDistance(vertices);

		for (int j=0; j<NUMBERofVERTICES; j++){
			toPrint[i][j]=minimumDistance[(j-i+NUMBERofVERTICES)%NUMBERofVERTICES];
		}
		delete [] minimumDistance;
	}

	cout << endl << endl << " Dijkstra" << endl;
	printVertices(toPrint);

	init(vertices, 0, example);
	cout << endl << " FloydWarshall" << endl;
	floydWarshall(vertices);
	printVertices(vertices);

	bool same=true;
	for (int i = 0; i < NUMBERofVERTICES; i++){
		for (int j = 0; j < NUMBERofVERTICES; j++){
			if(toPrint[i][j] != vertices[i][j]){ 
				same=false;
				break;
			}
		}
	}

	if(same) cout << "Dijkstra and FloydWarshall outputs are the same. OK!" << endl << endl;
	else cout << "Dijkstra and FloydWarshall outputs are not the same. ERROR!" << endl << endl;

	system ("pause");
	return 0;
}
