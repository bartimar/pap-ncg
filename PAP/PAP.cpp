#include "stdafx.h"


# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <ctime>
# include <omp.h>

using namespace std;

int NUMBERofVERTICES;
const int inf = INT_MAX;

int *dijkstraDistance (int** vertices);
void findNearest (int* minimumDistance, bool* connected, int& d, int& v);
void init (int**& vertices,int=0, int=1);
void updateMinimumDistance (int mv, bool* connected, int** vertices, int* minimumDistance);

void floydWarshall(int** vertices){

	for(int k=0; k<NUMBERofVERTICES; k++) {
		for(int i=0; i<NUMBERofVERTICES; i++){
			if(vertices[i][k] == inf) continue;
				for (int j=0; j<NUMBERofVERTICES; j++){
				if(vertices[k][j] == inf || i == j) continue;
				if(vertices[i][k] + vertices[k][j] < vertices[i][j]){
					vertices[i][j] = vertices[i][k] + vertices[k][j];
				}
			}
		}
	}

}

void printVertices(int** vertices) {

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

int *dijkstraDistance(int** vertices){
	bool *connected;
	int *minimumDistance;
	int myFirst = 1, myLast=NUMBERofVERTICES-1, distance, index;

	// start out with only node 0 connected to the tree
	connected = new bool[NUMBERofVERTICES];
	connected[0] = true;

	for(int i=1; i<NUMBERofVERTICES; i++)
		connected[i] = false;

	// initialize the minimum distance to the one-step distance
	minimumDistance = new int[NUMBERofVERTICES];

	for(int i=0; i<NUMBERofVERTICES; i++)
		minimumDistance[i] = vertices[0][i];


	for(int step=1; step<NUMBERofVERTICES; step++){
		
		findNearest(minimumDistance, connected, distance, index);

		if(distance < inf){
			connected[index] = true;
			updateMinimumDistance(index, connected, vertices, minimumDistance);
		}
	}

	delete [] connected;	// free memory

	return minimumDistance;
}

void findNearest(int* minimumDistance, bool *connected, int& distance, int& index){
	// output: 
	//	- int distance, the distance from node 0 to the nearest unconnected node in the range first to last
	//	- int index, the index of the nearest unconnected node in the range first to last.

	distance = inf;
	index = -1;

	for(int i=1; i<NUMBERofVERTICES; i++){
		if(!connected[i] && minimumDistance[i] < distance){
			distance = minimumDistance[i];
			index = i;
		}
	}
}

void alloc2Darray(int**& arr) {

	arr=new int*[NUMBERofVERTICES];
	
	for (int i = 0; i < NUMBERofVERTICES; i++)
		arr[i]=new int[NUMBERofVERTICES];
	{
	}
}

void dealloc2Darray(int**& arr) {

	
	for (int i = 0; i < NUMBERofVERTICES; i++)
	{
		delete arr[i];
	}

	delete [] arr;
	arr=NULL;
}

void init(int**& vertices,int shuf, int example){
	
	static bool initialized = false;

	if(!initialized) {
		for(int i=0; i<NUMBERofVERTICES; i++){
			for(int j=0; j<NUMBERofVERTICES; j++){
				if(i==j) vertices[i][i] = 0;
				else vertices[i][j] = inf;	// inicialization of all the other vertices to inf
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
		if(initialized) {
			int** newarr;
			alloc2Darray(newarr);
			for (int i = 0; i < NUMBERofVERTICES; i++)
			{
				for (int j = 0; j < NUMBERofVERTICES; j++)
				{		
					newarr[i][j]= vertices[(i+shuf)%NUMBERofVERTICES][(j+shuf)%NUMBERofVERTICES];
				}
			}
			dealloc2Darray(vertices);
			vertices=newarr;
			break;
		}
		for (int i = 0; i < NUMBERofVERTICES; i++)
		{
			for (int j = 0; j < NUMBERofVERTICES; j++)
			{				
				vertices[i][j]= (val=rand()%150) ? val : inf ;
			}
		}
		initialized=true;
	}


}

void updateMinimumDistance(int mainIndex, bool* connected, int** vertices, int* minimumDistance){
	for(int i=1;i< NUMBERofVERTICES; i++){
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
		NUMBERofVERTICES=15;
		break;
	default:
		cout<<"Bad parameter. Exit"<<endl;
		exit(1);
	}

}


void printInput(int** vertices) {

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
}

int _tmain(int argc, _TCHAR* argv[]){
	//CUDA - spoustet na 1,2,4,6,8,12,24
	int *minimumDistance;
	int** vertices=NULL,**toPrint=NULL;
	int example;
	srand((unsigned int)time(0));
	
	initExample(example);
	alloc2Darray(vertices);
	alloc2Darray(toPrint);
	// inicialization of data
	init(vertices, 0, example);	

	// print input
	printInput(vertices);

	//launch Dijkstra
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

	
	printInput(vertices);

	//launch FloydWarshall
	//init(vertices, 0, example);
	cout << endl << " FloydWarshall" << endl;
	floydWarshall(vertices);
	printVertices(vertices);

	//check if the outputs were the same
	bool same=true;
	for (int i = 0; i < NUMBERofVERTICES; i++){
		for (int j = 0; j < NUMBERofVERTICES; j++){
			if(toPrint[i][j] != vertices[i][j]){ 
				same=false;
				//cout<< "Error at vertices["<< i<<"]["<< j<<"]" <<endl;
				//break;
			}
		}
	}

	if(same) cout <<"NoV="<< NUMBERofVERTICES<<": Dijkstra and FloydWarshall outputs are the same. OK!" << endl << endl;
	else cout << "NoV="<< NUMBERofVERTICES<<": Dijkstra and FloydWarshall outputs are not the same. ERROR!" << endl << endl;

	
	dealloc2Darray(vertices);
	dealloc2Darray(toPrint);

	system ("pause");
	return 0;
}
