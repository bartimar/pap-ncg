#pragma once

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

static void HandleError( cudaError_t err, const char *file, int line ) {
 if (err != cudaSuccess) { 
 printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line ); 
 exit( EXIT_FAILURE ); 
}} 
 
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 

#define inf INT_MAX

__device__ int *dijkstraDistance_GPU (int* vertices, int shuf, int NUMBERofVERTICES);
__device__ void findNearest_GPU (int* minimumDistance, bool* connected, int& d, int& v,int, int NUMBERofVERTICES);
__device__ void updateMinimumDistance_GPU (int mv, bool* connected, int** vertices, int* minimumDistance, int NUMBERofVERTICES);
__global__ void dijkstra_GPU(int* vertices, int blockId, int NUMBERofVERTICES);

int *dijkstraDistance (int** vertices, int shuf,int NUMBERofVERTICES);
void findNearest (int* minimumDistance, bool* connected, int& d, int& v,int,int NUMBERofVERTICES);
void updateMinimumDistance (int mv, bool* connected, int** vertices, int* minimumDistance,int NUMBERofVERTICES);
void dijkstra(int** vertices, int** toPrint,int NUMBERofVERTICES);

void floydWarshall_GPU(int *HostGraph, const int NUMBERofVERTICES);
void floydWarshall(int** vertices,int num_threads);
__global__ void floydWarshall_GPU_kernel(int k, int *Graph,int NUMBERofVERTICES);


void init (int**& vertices,int=0, int=1);