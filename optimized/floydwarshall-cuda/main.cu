/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <chrono>
#include <cuda.h>

#define MAXDISTANCE    (200)

/**
 * Returns the lesser of the two unsigned integers a and b
 */
unsigned int minimum(unsigned int a, unsigned int b) 
{
  return (b < a) ? b : a;
}

/**
 * Reference CPU implementation of FloydWarshall PathFinding
 * for performance comparison
 * @param pathDistanceMatrix Distance between nodes of a graph
 * @param intermediate node between two nodes of a graph
 * @param number of nodes in the graph
 */
void floydWarshallCPUReference(unsigned int * pathDistanceMatrix,
    unsigned int * pathMatrix, unsigned int numNodes)
{
  unsigned int distanceYtoX, distanceYtoK, distanceKtoX, indirectDistance;

  /*
   * pathDistanceMatrix is the adjacency matrix(square) with
   * the dimension equal to the number of nodes in the graph.
   */
  unsigned int width = numNodes;
  unsigned int yXwidth;

  /*
   * for each intermediate node k in the graph find the shortest distance between
   * the nodes i and j and update as
   *
   * ShortestPath(i,j,k) = min(ShortestPath(i,j,k-1), ShortestPath(i,k,k-1) + ShortestPath(k,j,k-1))
   */
  for(unsigned int k = 0; k < numNodes; ++k)
  {
    for(unsigned int y = 0; y < numNodes; ++y)
    {
      yXwidth =  y*numNodes;
      for(unsigned int x = 0; x < numNodes; ++x)
      {
        distanceYtoX = pathDistanceMatrix[yXwidth + x];
        distanceYtoK = pathDistanceMatrix[yXwidth + k];
        distanceKtoX = pathDistanceMatrix[k * width + x];

        indirectDistance = distanceYtoK + distanceKtoX;

        if(indirectDistance < distanceYtoX)
        {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x]         = k;
        }
      }
    }
  }
}


/*!
 * The floyd Warshall algorithm is a multipass algorithm
 * that calculates the shortest path between each pair of
 * nodes represented by pathDistanceBuffer.
 *
 * In each pass a node k is introduced and the pathDistanceBuffer
 * which has the shortest distance between each pair of nodes
 * considering the (k-1) nodes (that are introduced in the previous
 * passes) is updated such that
 *
 * ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) + ShortestPath(k,y,k-1))
 * where x and y are the pair of nodes between which the shortest distance
 * is being calculated.
 *
 * pathBuffer stores the intermediate nodes through which the shortest
 * path goes for each pair of nodes.
 *
 * numNodes is the number of nodes in the graph.
 *
 * for more detailed explaination of the algorithm kindly refer to the document
 * provided with the sample
 */

#define TILE_SIZE 16

__global__ void floydWarshallPhase1(
    unsigned int *__restrict__ pathDistanceBuffer,
    unsigned int *__restrict__ pathBuffer,
    const unsigned int numNodes,
    const unsigned int pivot)
{
  __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];
  int x = pivot * TILE_SIZE + threadIdx.x;
  int y = pivot * TILE_SIZE + threadIdx.y;
  tile[threadIdx.y][threadIdx.x] = pathDistanceBuffer[y * numNodes + x];
  __syncthreads();

  for (int k = 0; k < TILE_SIZE; k++) {
    unsigned int indirect = tile[threadIdx.y][k] + tile[k][threadIdx.x];
    __syncthreads();
    if (indirect < tile[threadIdx.y][threadIdx.x]) {
      tile[threadIdx.y][threadIdx.x] = indirect;
      pathBuffer[y * numNodes + x] = pivot * TILE_SIZE + k;
    }
    __syncthreads();
  }
  pathDistanceBuffer[y * numNodes + x] = tile[threadIdx.y][threadIdx.x];
}

__global__ void floydWarshallPhase2(
    unsigned int *__restrict__ distance,
    unsigned int *__restrict__ path,
    const unsigned int numNodes,
    const unsigned int pivot)
{
  int tile_id = blockIdx.x;
  if (tile_id == pivot) return;

  __shared__ unsigned int pivot_tile[TILE_SIZE][TILE_SIZE];
  __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];
  int tx = threadIdx.x, ty = threadIdx.y;
  pivot_tile[ty][tx] = distance[(pivot*TILE_SIZE + ty) * numNodes + pivot*TILE_SIZE + tx];

  bool row = blockIdx.y == 0;
  int x = (row ? tile_id : pivot) * TILE_SIZE + tx;
  int y = (row ? pivot : tile_id) * TILE_SIZE + ty;
  tile[ty][tx] = distance[y * numNodes + x];
  __syncthreads();

  for (int k = 0; k < TILE_SIZE; k++) {
    unsigned int indirect = row
      ? pivot_tile[ty][k] + tile[k][tx]
      : tile[ty][k] + pivot_tile[k][tx];
    __syncthreads();
    if (indirect < tile[ty][tx]) {
      tile[ty][tx] = indirect;
      path[y * numNodes + x] = pivot * TILE_SIZE + k;
    }
    __syncthreads();
  }
  distance[y * numNodes + x] = tile[ty][tx];
}

__global__ void floydWarshallPhase3(
    unsigned int *__restrict__ distance,
    unsigned int *__restrict__ path,
    const unsigned int numNodes,
    const unsigned int pivot)
{
  if (blockIdx.x == pivot || blockIdx.y == pivot) return;

  __shared__ unsigned int column_tile[TILE_SIZE][TILE_SIZE];
  __shared__ unsigned int row_tile[TILE_SIZE][TILE_SIZE];
  int tx = threadIdx.x, ty = threadIdx.y;
  int x = blockIdx.x * TILE_SIZE + tx;
  int y = blockIdx.y * TILE_SIZE + ty;

  column_tile[ty][tx] = distance[y * numNodes + pivot*TILE_SIZE + tx];
  row_tile[ty][tx] = distance[(pivot*TILE_SIZE + ty) * numNodes + x];
  __syncthreads();

  unsigned int best = distance[y * numNodes + x];
  int best_k = -1;
  #pragma unroll
  for (int k = 0; k < TILE_SIZE; k++) {
    unsigned int indirect = column_tile[ty][k] + row_tile[k][tx];
    if (indirect < best) {
      best = indirect;
      best_k = pivot * TILE_SIZE + k;
    }
  }
  distance[y * numNodes + x] = best;
  if (best_k >= 0) path[y * numNodes + x] = best_k;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
    return 1;
  }
  // There are three required command-line arguments
  unsigned int numNodes = atoi(argv[1]);
  unsigned int numIterations = atoi(argv[2]);
  unsigned int blockSize = atoi(argv[3]);

  // The blocked implementation operates on fixed 16x16 tiles.
  if(numNodes % TILE_SIZE != 0) {
    numNodes = (numNodes / TILE_SIZE + 1) * TILE_SIZE;
  }

  // allocate and init memory used by host
  unsigned int* pathMatrix = NULL;
  unsigned int* pathDistanceMatrix = NULL;
  unsigned int* verificationPathDistanceMatrix = NULL;
  unsigned int* verificationPathMatrix = NULL;
  unsigned int matrixSizeBytes;

  matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);
  pathDistanceMatrix = (unsigned int *) malloc(matrixSizeBytes);
  assert (pathDistanceMatrix != NULL);

  pathMatrix = (unsigned int *) malloc(matrixSizeBytes);
  assert (pathMatrix != NULL) ;

  // input must be initialized; otherwise host and device results may be different
  srand(2);
  for(unsigned int i = 0; i < numNodes; i++)
    for(unsigned int j = 0; j < numNodes; j++)
    {
      int index = i*numNodes + j;
      pathDistanceMatrix[index] = rand() % (MAXDISTANCE + 1);
    }
  for(unsigned int i = 0; i < numNodes; ++i)
  {
    unsigned int iXWidth = i * numNodes;
    pathDistanceMatrix[iXWidth + i] = 0;
  }

  /*
   * pathMatrix is the intermediate node from which the path passes
   * pathMatrix(i,j) = k means the shortest path from i to j
   * passes through an intermediate node k
   * Initialized such that pathMatrix(i,j) = i
   */
  for(unsigned int i = 0; i < numNodes; ++i)
  {
    for(unsigned int j = 0; j < i; ++j)
    {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }

  verificationPathDistanceMatrix = (unsigned int *) malloc(numNodes * numNodes * sizeof(int));
  assert (verificationPathDistanceMatrix != NULL);

  verificationPathMatrix = (unsigned int *) malloc(numNodes * numNodes * sizeof(int));
  assert(verificationPathMatrix != NULL);

  memcpy(verificationPathDistanceMatrix, pathDistanceMatrix,
      numNodes * numNodes * sizeof(int));
  memcpy(verificationPathMatrix, pathMatrix, numNodes*numNodes*sizeof(int));

  unsigned int numPasses = numNodes;

  unsigned int globalThreads[2] = {numNodes, numNodes};
  unsigned int localThreads[2] = {blockSize, blockSize};

  if((unsigned int)(localThreads[0] * localThreads[0]) >256)
  {
    blockSize = 16;
    localThreads[0] = blockSize;
    localThreads[1] = blockSize;
  }

  dim3 grids( globalThreads[0]/localThreads[0], globalThreads[1]/localThreads[1]);
  dim3 threads (localThreads[0],localThreads[1]);

  unsigned int *pathDistanceBuffer, *pathBuffer;
  cudaMalloc((void**)&pathDistanceBuffer, matrixSizeBytes);
  cudaMalloc((void**)&pathBuffer, matrixSizeBytes);

  float total_time = 0.f;

  // copy the matrix from a host to a device "iterations" times,
  // but copy the result from a device to a host once
  for (unsigned int n = 0; n < numIterations; n++) {
    /*
     * The floyd Warshall algorithm is a multipass algorithm
     * that calculates the shortest path between each pair of
     * nodes represented by pathDistanceBuffer.
     *
     * In each pass a node k is introduced and the pathDistanceBuffer
     * which has the shortest distance between each pair of nodes
     * considering the (k-1) nodes (that are introduced in the previous
     * passes) is updated such that
     *
     * ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) + ShortestPath(k,y,k-1))
     * where x and y are the pair of nodes between which the shortest distance
     * is being calculated.
     *
     * pathBuffer stores the intermediate nodes through which the shortest
     * path goes for each pair of nodes.
     */

    cudaMemcpy(pathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    const unsigned int numTiles = numPasses / TILE_SIZE;
    dim3 blockedThreads(TILE_SIZE, TILE_SIZE);
    dim3 phase2Grid(numTiles, 2);
    dim3 phase3Grid(numTiles, numTiles);
    for(unsigned int i = 0; i < numTiles; i++)
    {
      floydWarshallPhase1<<<1, blockedThreads>>>(pathDistanceBuffer, pathBuffer, numNodes, i);
      floydWarshallPhase2<<<phase2Grid, blockedThreads>>>(pathDistanceBuffer, pathBuffer, numNodes, i);
      floydWarshallPhase3<<<phase3Grid, blockedThreads>>>(pathDistanceBuffer, pathBuffer, numNodes, i);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / numIterations);

  cudaMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost);
  cudaFree(pathDistanceBuffer);
  cudaFree(pathBuffer);

  // verify
  floydWarshallCPUReference(verificationPathDistanceMatrix, verificationPathMatrix, numNodes);
  if(memcmp(pathDistanceMatrix, verificationPathDistanceMatrix, matrixSizeBytes) == 0)
  {
    printf("PASS\n");
  }
  else
  {
    printf("FAIL\n");
    if (numNodes <= 8) 
    {
      for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++)
          printf("host: %u ", verificationPathDistanceMatrix[i*numNodes+j]);
        printf("\n");
      }
      for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++)
          printf("device: %u ", pathDistanceMatrix[i*numNodes+j]);
        printf("\n");
      }
    }
  }

  free(pathDistanceMatrix);
  free(pathMatrix);
  free(verificationPathDistanceMatrix);
  free(verificationPathMatrix);
  return 0;
}
