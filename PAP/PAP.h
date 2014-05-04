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


#define inf INT_MAX

__device__ int *dijkstraDistance (int** vertices, int shuf, int NUMBERofVERTICES);
__device__ void findNearest (int* minimumDistance, bool* connected, int& d, int& v,int, int NUMBERofVERTICES);
__device__ void updateMinimumDistance (int mv, bool* connected, int** vertices, int* minimumDistance, int NUMBERofVERTICES);
__global__  void dijkstra(int** vertices, int** toPrint, int example, int NUMBERofVERTICES);

void floydWarshall_GPU(int *HostGraph, const int NUMBERofVERTICES);
void floydWarshall(int** vertices,int num_threads);
__global__ void floydWarshall_GPU_kernel(int k, int *G,int N);
__global__ void _Wake_GPU(int reps);




void init (int**& vertices,int=0, int=1);