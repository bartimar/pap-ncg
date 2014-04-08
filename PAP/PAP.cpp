# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <ctime>
#include <limits.h>
# include <omp.h>

using namespace std;

int NUMBERofVERTICES;
bool *connected;
	
const int inf = INT_MAX;
int *dijkstraDistance (int** vertices);
void findNearest (int* minimumDistance, bool* connected, int& d, int& v,int);
void init (int**& vertices,int=0, int=1);
void updateMinimumDistance (int mv, bool* connected, int** vertices, int* minimumDistance);

void floydWarshall(int** vertices,int num_threads){

	double start,end;
	start=omp_get_wtime();
	for(int k=0; k<NUMBERofVERTICES; k++) {
	
		
		#pragma omp master
		omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(num_threads); // Use 4 threads for all consecutive parallel regions
		int i,j;

		#pragma omp parallel for private(i,j), shared(k)
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

	end=omp_get_wtime();
	

	cout<< "Time Warshall: "<< end-start <<endl;
}

void printVertices(int** vertices) {
#ifdef DEBUG
	cout<< "    ";
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
#endif
}

int *dijkstraDistance(int** vertices,int shuf){
	int *minimumDistance;
	int  distance, index;

	for(int i=0; i<NUMBERofVERTICES; i++)
		connected[i] = false;

	connected[shuf] = true;

	// initialize the minimum distance to the one-step distance
	minimumDistance = new int[NUMBERofVERTICES];

	for(int i=0; i<NUMBERofVERTICES; i++)
		minimumDistance[i] = vertices[shuf][i];


	for(int step=1; step<NUMBERofVERTICES; step++){
		
		findNearest(minimumDistance, connected, distance, index,shuf);

		if(distance < inf){
			connected[index] = true;
			updateMinimumDistance(index, connected, vertices, minimumDistance);
		}
	}

	return minimumDistance;
}

void findNearest(int* minimumDistance, bool *connected, int& distance, int& index,int shuf){
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

void alloc2Darray(int**& arr) {

	arr=new int*[NUMBERofVERTICES];
	
	for (int i = 0; i < NUMBERofVERTICES; i++)
		arr[i]=new int[NUMBERofVERTICES];
}

void dealloc2Darray(int**& arr) {
	
	for (int i = 0; i < NUMBERofVERTICES; i++)
		delete arr[i];
	
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
		// start out with only node shuf connected to the tree
	connected = new bool[NUMBERofVERTICES];

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
		if(initialized) break;
		for (int i = 0; i < NUMBERofVERTICES; i++)
			for (int j = 0; j < NUMBERofVERTICES; j++)
				vertices[i][j]= (val=rand()%150) ? val : inf ;

		initialized=true;
	}


}

void updateMinimumDistance(int mainIndex, bool* connected, int** vertices, int* minimumDistance){
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

	cout<<"Enter number of threads: "<<endl;
	cin>>num_threads;
	
}
void printInput(int** vertices) {
#ifdef DEBUG
	cout << "Input matrix of distances" << endl << endl;

	for(int k=0; k<NUMBERofVERTICES; k++){
		for(int j=0; j<NUMBERofVERTICES; j++){
			if(vertices[k][j] == inf) cout << setw(6) << "Inf";
			else cout << setw(6) <<  vertices[k][j];
		}
		cout << endl;
	}
#endif
}

void dijkstra(int** vertices, int** toPrint, int example,int num_threads) {
	int *minimumDistance,i;

	double start,end;
	start=omp_get_wtime();

#pragma omp master
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(num_threads); // Use x threads for all consecutive parallel regions
#pragma omp parallel for private(i), shared(vertices)
	for(i = 0; i < NUMBERofVERTICES; i++){

		minimumDistance = dijkstraDistance(vertices,i);

		for (int j=0; j<NUMBERofVERTICES; j++){
			toPrint[i][j]=minimumDistance[j];
		}
		delete [] minimumDistance;
	}


	end=omp_get_wtime();	

	cout<< "Time Dijkstra: "<< end-start <<endl;

}

int main(int argc, char** argv){
	//CUDA - spoustet na 1,2,4,6,8,12,24
	int** vertices=NULL,**toPrint=NULL;
	int example,num_threads;
	int i;
	srand((unsigned int)time(NULL));
	
	//initExample(example);
	example=0;
	NUMBERofVERTICES=5000;
	//initThreads(num_threads);
	num_threads=4;


	alloc2Darray(vertices);
	alloc2Darray(toPrint);
	// inicialization of data
	init(vertices, 0, example);	

	// print input
	printInput(vertices);

	//launch Dijkstra
	dijkstra(vertices,toPrint,example,num_threads);
	
	//cout << endl << endl << " Dijkstra" << endl;
	printVertices(toPrint);

	
	printInput(vertices);

	//launch FloydWarshall
	//cout << endl << " FloydWarshall" << endl;
	floydWarshall(vertices,num_threads);
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

	delete [] connected;

	system ("pause");
	return 0;
}
