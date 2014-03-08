// PAP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <ctime>
# include <omp.h>

using namespace std;

# define NUMBERofVERTICES 6

int *dijkstraDistance (int vertices[NUMBERofVERTICES][NUMBERofVERTICES]);
void findNearest (int s, int e, int minimumDistance[NUMBERofVERTICES], bool connected[NUMBERofVERTICES], int *d, int *v);
void init (int vertices[NUMBERofVERTICES][NUMBERofVERTICES],int=0);
void updateMinimumDistance (int s, int e, int mv, bool connected[NUMBERofVERTICES], int vertices[NUMBERofVERTICES][NUMBERofVERTICES], int minimumDistance[NUMBERofVERTICES]);

void floydWarshall(int vertices[NUMBERofVERTICES][NUMBERofVERTICES]){
	int i, j, k;
	for(k=0; k<NUMBERofVERTICES; k++) {
		for(i=0; i<NUMBERofVERTICES; i++){
			for (j=0; j<NUMBERofVERTICES; j++){
				/* If i and j are different nodes and if 
				the paths between i and k and between
				k and j exist, do */
				if((vertices[i][k]*vertices[k][j] != 0) && (i != j)){
					/* See if you can't get a shorter path
					between i and j by interspacing
					k somewhere along the current
					path */
					if((vertices[i][k] + vertices[k][j] < vertices[i][j]) ||(vertices[i][j] == 0)){
						vertices[i][j] = vertices[i][k] + vertices[k][j];
					}
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
			cout << setw(4) <<  vertices[i][j];
		}
		cout<< endl;
	}
	cout<<endl;

}

int *dijkstraDistance(int vertices[NUMBERofVERTICES][NUMBERofVERTICES]){
	//    We essentially build a tree.  We start with only node 0 connected
	//    to the tree, and this is indicated by setting CONNECTED[0] = TRUE.

	//    We initialize minimumDistance[I] to the one step distance from node 0 to node I.

	//    Now we search among the unconnected nodes for the node MV whose minimum
	//    distance is smallest, and connect it to the tree.  For each remaining
	//    unconnected node I, we check to see whether the distance from 0 to MV
	//    to I is less than that recorded in minimumDistance[I], and if so, we can reduce
	//    the distance.
	bool *connected;
	int i, *minimumDistance, inf = 2147483647;
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

	int i, inf=2147483647;
	*distance = inf;
	*index = -1;

	for(i=first; i<=last; i++){
		if(!connected[i] && minimumDistance[i] < *distance){
			*distance = minimumDistance[i];
			*index = i;
		}
	}
}

void init(int vertices[NUMBERofVERTICES][NUMBERofVERTICES],int shuf){
	//    The graph uses 6 nodes, and has the following diagram and
	//    distance matrix:

	//    N0--15--N2-100--N3           0   40   15  Inf  Inf  Inf
	//      \      |     /            40    0   20   10   25    6
	//       \     |    /             15   20    0  100  Inf  Inf
	//        40  20  10             Inf   10  100    0  Inf  Inf
	//          \  |  /              Inf   25  Inf  Inf    0    8
	//           \ | /               Inf    6  Inf  Inf    8    0
	//            N1
	//            / \
	//           /   \
	//          6    25
	//         /       \
	//        /         \
	//      N5----8-----N4

	//    Output, int vertices[NUMBERofVERTICES][NUMBERofVERTICES], the distance of the direct link between
	//    nodes I and J.

	int i, j, inf = 2147483647;

	if(shuf==-1) {
		inf=0;
		shuf=0;
	}
	for(i=0; i<NUMBERofVERTICES; i++){
		for(j=0; j<NUMBERofVERTICES; j++){
			if(i==j){					// the same vertices, distance = 0			
				vertices[i][i] = 0;

			}else{
				vertices[i][j] = inf;	// inicialization of all the other vertices to inf
			}
		}
	}

	// our task
	vertices[(0+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = vertices[(1+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 40;
	vertices[(0+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = vertices[(2+shuf)%NUMBERofVERTICES][(0+shuf)%NUMBERofVERTICES] = 15;
	vertices[(1+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = vertices[(2+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 20;
	vertices[(1+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = vertices[(3+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 10;
	vertices[(1+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = vertices[(4+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 25;
	vertices[(2+shuf)%NUMBERofVERTICES][(3+shuf)%NUMBERofVERTICES] = vertices[(3+shuf)%NUMBERofVERTICES][(2+shuf)%NUMBERofVERTICES] = 100;
	vertices[(1+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = vertices[(5+shuf)%NUMBERofVERTICES][(1+shuf)%NUMBERofVERTICES] = 6;
	vertices[(4+shuf)%NUMBERofVERTICES][(5+shuf)%NUMBERofVERTICES] = vertices[(5+shuf)%NUMBERofVERTICES][(4+shuf)%NUMBERofVERTICES] = 8;
}

void updateMinimumDistance(int first, int last, int mainIndex, bool connected[NUMBERofVERTICES], int vertices[NUMBERofVERTICES][NUMBERofVERTICES], int minimumDistance[NUMBERofVERTICES]){
	int i, inf = 2147483647;

	for(i=first; i <= last; i++){
		if(!connected[i]){

			if(vertices[mainIndex][i] < inf){

				if(minimumDistance[mainIndex] + vertices[mainIndex][i] < minimumDistance[i]){
					minimumDistance[i] = minimumDistance[mainIndex] + vertices[mainIndex][i];
				}
			}
		}
	}
}



int _tmain(int argc, _TCHAR* argv[])
{
	int *minimumDistance, inf = 2147483647;
	int vertices[NUMBERofVERTICES][NUMBERofVERTICES];
	int toPrint[NUMBERofVERTICES][NUMBERofVERTICES];

	// inicialization of data
	init(vertices,0);	

	// print input
	cout << "Input matrix of distances\n\n";

	for(int k=0; k<NUMBERofVERTICES; k++){
		for(int j=0; j<NUMBERofVERTICES; j++){
			if(vertices[k][j] == inf){
				cout << setw(6) << "Inf";

			}else{
				cout << setw(6) <<  vertices[k][j];
			}
		}
		cout << "\n";
	}

	for(int i = 0; i < NUMBERofVERTICES; i++){

		// inicialization of data
		init(vertices,-i+NUMBERofVERTICES);	


		minimumDistance = dijkstraDistance(vertices);
		//WTF
		//// print the results
		//cout << "\nMinimum distances from node "<< i<< endl;
		//cout << " -Dijkstra" << endl;

		for (int j=0; j<NUMBERofVERTICES; j++){
		//	cout << setw(4) << j << "  " << setw(4) << minimumDistance[(j-i+NUMBERofVERTICES)%NUMBERofVERTICES] << "\n";
		toPrint[i][j]=minimumDistance[(j-i+NUMBERofVERTICES)%NUMBERofVERTICES];
		}
		delete [] minimumDistance;

	}

	cout << "\nDijkstra" << endl;
	printVertices(toPrint);
	init(vertices,-1);
	cout << "\nFloydWarshall" << endl;
	floydWarshall(vertices);
	printVertices(vertices);

	system ("pause");
	return 0;
}

