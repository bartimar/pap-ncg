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
void init (int vertices[NUMBERofVERTICES][NUMBERofVERTICES]);
void updateMinimumDistance (int s, int e, int mv, bool connected[NUMBERofVERTICES], int vertices[NUMBERofVERTICES][NUMBERofVERTICES], int minimumDistance[NUMBERofVERTICES]);

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

void findNearest (int first, int last, int minimumDistance[NUMBERofVERTICES], bool connected[NUMBERofVERTICES], int *distance, int *index){
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

void init (int vertices[NUMBERofVERTICES][NUMBERofVERTICES])
{
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
	vertices[0][1] = vertices[1][0] = 40;
	vertices[0][2] = vertices[2][0] = 15;
	vertices[1][2] = vertices[2][1] = 20;
	vertices[1][3] = vertices[3][1] = 10;
	vertices[1][4] = vertices[4][1] = 25;
	vertices[2][3] = vertices[3][2] = 100;
	vertices[1][5] = vertices[5][1] = 6;
	vertices[4][5] = vertices[5][4] = 8;
}

void updateMinimumDistance (int first, int last, int mainIndex, bool connected[NUMBERofVERTICES], int vertices[NUMBERofVERTICES][NUMBERofVERTICES], int minimumDistance[NUMBERofVERTICES]){
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
	int i, j, *minimumDistance, inf = 2147483647;
	int vertices[NUMBERofVERTICES][NUMBERofVERTICES];

	// inicialization of data
	init(vertices);	

	// print input
	cout << "Input matrix of distances\n\n";

	for(i=0; i<NUMBERofVERTICES; i++){
		for(j=0; j<NUMBERofVERTICES; j++){
			if(vertices[i][j] == inf){
				cout << setw(6) << "Inf";

			}else{
				cout << setw(6) <<  vertices[i][j];
			}
		}
		cout << "\n";
	}

	minimumDistance = dijkstraDistance(vertices);

	// print the results
	cout << "\nMinimum distances from node 0\n\n";

	for (i=0; i<NUMBERofVERTICES; i++){
		cout << setw(4) << i << "  " << setw(4) << minimumDistance[i] << "\n";
	}

	delete [] minimumDistance;		// free memory

	system ("pause");
	return 0;
}

