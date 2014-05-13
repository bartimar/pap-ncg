#include "PAP.h"


int NUMBERofVERTICES;




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
			for (int j = 0; j < NUMBERofVERTICES; j++){
				if(i==j) continue;
				D_G[i*NUMBERofVERTICES+j]=vertices[i][j]= (val=rand()%150) ? val : inf ;
			}

		initialized=true;
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


int main(int argc, char** argv){
	//CUDA - spoustet na 1,2,4,6,8,12,24
	int **vertices=NULL,**toPrint=NULL;
	int example,num_threads;
	srand((unsigned int)time(NULL));
	int *H_G;
	cudaEvent_t start,start2, stop,stop2; 
	float elapsedTime,elapsedTime2; 

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
	
	alloc2Darray(vertices);
	alloc2Darray(toPrint);
	//int * minimumDistance;
	HANDLE_ERROR ( cudaHostAlloc((int**)&H_G, NUMBERofVERTICES*NUMBERofVERTICES*sizeof(int),cudaHostAllocDefault));	
	//HANDLE_ERROR ( cudaHostAlloc((int**)&minimumDistance, NUMBERofVERTICES*NUMBERofVERTICES*sizeof(int),cudaHostAllocDefault));	
	HANDLE_ERROR ( cudaMemset(H_G,0,NUMBERofVERTICES*NUMBERofVERTICES*sizeof(int)));	

	//int* D_G;
	//HANDLE_ERROR ( cudaMalloc((int**)&D_G, NUMBERofVERTICES*NUMBERofVERTICES*sizeof(int)));	
	
	//cudaMemcpy(D_G,H_G,sizeof(int)*NUMBERofVERTICES*NUMBERofVERTICES,cudaMemcpyHostToDevice);
	//alloc2Darray(toPrint);
	// inicialization of data
	init(vertices, 0, example, H_G);	
	
	cout << "   Dijkstra" << endl;
	cudaEventCreate( &start2 ) ; 
	cudaEventCreate( &stop2 ) ; 
	cudaEventRecord( start2, 0 );

	////////launch Dijkstra
	dijkstra(vertices,toPrint,NUMBERofVERTICES);
	cudaEventRecord( stop2, 0 );
	cudaEventSynchronize( stop2 ) ; 
	cudaEventElapsedTime( &elapsedTime2, start2, stop2 ); 
	cout << "CPU time taken: "<< elapsedTime2 <<  " ms" << endl; 
	
	//printVertices(toPrint);
	/*for (int i = 0; i < NUMBERofVERTICES*NUMBERofVERTICES; i++)
	{
		minimumDistance[i]=H_G[i];
	}*/

	cout << "   Dijkstra_GPU" << endl;
	cudaEventCreate( &start2 ) ; 
	cudaEventCreate( &stop2 ) ; 
	cudaEventRecord( start2, 0 );
	
	dim3 dimGrid((NUMBERofVERTICES+256-1)/256,NUMBERofVERTICES);

	////////launch Dijkstra
	/*for (int i = 0; i < NUMBERofVERTICES/8; i++)
	{*/
	dijkstra_GPU<<<8,NUMBERofVERTICES/8>>>(H_G,0, NUMBERofVERTICES);
	
	/*}*/
	
	cudaThreadSynchronize(); 

	cudaEventRecord( stop2, 0 );
	cudaEventSynchronize( stop2 ) ; 
	cudaEventElapsedTime( &elapsedTime2, start2, stop2 ); 
	cout << "GPU time taken: "<< elapsedTime2 <<  " ms" << endl; 
	
	//printVertices_GPU(minimumDistance);
	//int failI,failJ;
	cout << endl << "Comparing results..." << endl;
	bool same=true;
	//for (int i = 0; i < NUMBERofVERTICES; i++){
	//	for (int j = 0; j < NUMBERofVERTICES; j++){
	//		if(toPrint[i][j]!=minimumDistance[i*NUMBERofVERTICES+j]){
	//		//failI=i;
	//		//failJ=j;
	//			same=false;
	//			//cout << "Outputs are NOT the same... fail at [" << failI<< "]["<< failJ<<"] ... "<< vertices[i][j]<<"!="<< minimumDistance[j*NUMBERofVERTICES+i]<< ""<< endl;
	//		}
	//	}
	//}
	if(same) cout<< "Outputs are the same."<<endl;
	else cout << "Outputs are NOT the same..."<< endl;

	cout << "   FloydWarshall_GPU" << endl;
	cudaEventCreate( &start ) ; 
	cudaEventCreate( &stop ) ; 
	cudaEventRecord( start, 0 );

	floydWarshall_GPU(H_G, NUMBERofVERTICES);
	floydWarshall(vertices, NUMBERofVERTICES);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop ) ; 
	cudaEventElapsedTime( &elapsedTime, start, stop ); 

	cout  << "GPU time taken: "<< elapsedTime <<  " ms" << endl; 

	cudaThreadSynchronize(); 

	//printVertices_GPU(H_G);
	//cout << "   FloydWarshall" << endl;
	//printVertices(vertices);
	/*check if the outputs were the same*/
	/*cout << endl << "Comparing results..." << endl;
	bool same=true;
	for (int i = 0; i < NUMBERofVERTICES*NUMBERofVERTICES; i++){
		if(H_G[i]!=minimumDistance[i]) same=false;
	}
	cout<< (same ? "Outputs are the same." : "Outputs are NOT the same")<< endl;*/
	////////

	//if(same) cout <<"NoV="<< NUMBERofVERTICES<<": Dijkstra and FloydWarshall outputs are the same. OK!" << endl << endl;
	//else cout << "NoV="<< NUMBERofVERTICES<<": Dijkstra and FloydWarshall outputs are not the same. ERROR!" << endl << endl;


	dealloc2Darray(vertices);
	HANDLE_ERROR ( cudaFreeHost(H_G));
	//	dealloc2Darray(toPrint);

	//system ("pause");
	return 0;
}