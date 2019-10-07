#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilities.h"
#include "kernel.h"
#include "svd3.h"
#include <thrust/reduce.h>
//#include <glm/gtx/string_cast.hpp>
using namespace std;

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
int sourceSize;
int targetSize;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *devCorrespond;
glm::vec3 *devTempSource;
glm::mat3 *devMult;
// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__global__ void kernResetVec3Buffer(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}


__global__ void calculateCorrespondPoint(int sourceSize,int targetSize, glm::vec3 *devPos, glm::vec3 *correspond) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= sourceSize)
		return;

	float min_dist = glm::distance(devPos[index],devPos[sourceSize]);
	int id = 0;
	float dist;
	for (int i = 1; i < targetSize; i++) {
		dist = glm::distance(devPos[index], devPos[i+sourceSize]);
		if (dist < min_dist) {
			min_dist = dist;
			id = i;
		}
	}
	correspond[index] = devPos[id+sourceSize];
}

__global__ void meanCentrePoints(int N, glm::vec3 *tmpSource, glm::vec3 *correspond, glm::vec3 meanSource, glm::vec3 meanCorrespond) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	tmpSource[index] -= meanSource;
	correspond[index] -= meanCorrespond;

}

/**
* Initialize memory, update some globals
*/
void scanMatchingICP::initSimulation(vector<glm::vec3>& source, vector<glm::vec3>& target) {
  
  numObjects = source.size() + target.size();
  sourceSize = source.size();
  targetSize = target.size();
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, numObjects * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, numObjects * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&devCorrespond, sourceSize * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc devCorrespond failed!");

  cudaMalloc((void**)&devTempSource, sourceSize * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc devTempSource failed!");

  cudaMalloc((void**)&devMult, sourceSize * sizeof(glm::mat3));
  checkCUDAErrorWithLine("cudaMalloc devTempSource failed!");


  //cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec4));
  //checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // copy both scene and target to output points
   cudaMemcpy(dev_pos, &source[0], source.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
   cudaMemcpy(&dev_pos[source.size()], &target[0], target.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

   kernResetVec3Buffer << <dim3((source.size() + blockSize - 1) / blockSize), blockSize >> > (source.size(), dev_vel1, glm::vec3(1, 1, 1));
   kernResetVec3Buffer << <dim3((target.size() + blockSize - 1) / blockSize), blockSize >> > (target.size(), &dev_vel1[source.size()], glm::vec3(1, 1, 0));

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}


__global__ void outerProduct(int sourceSize, glm::vec3 *source, glm::vec3 *target,glm::mat3 *out) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= sourceSize)
		return;

	out[index] = glm::outerProduct(source[index], target[index]);
}

__global__ void kernMatrixMultiplication(glm::vec3 *M, glm::vec3 *N, float *Out, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("The values of m , n and k are :%d , %d %d \n", m, n , k);
	//printf("The values of row and col are: %d & %d \n", row, col);
	float sum = 0;
	float a, b;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			a = (row == 0 ? N[i].x : row == 1 ? N[i].y : N[i].z);
			b = (col == 0 ? M[i].x : col == 1 ? M[i].y : M[i].z);
			sum += a * b;
			//printf("hello the value of Sum is : %0.3f\n",sum);
		}
		//printf("The values are %d & %d \n", row, col);
		Out[row*k + col] = sum;
		//printf("The value is: %0.2f \n", Out[row*k + col]);
	}

}

__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < cols && idy < rows)
	{
		unsigned int pos = idy * cols + idx;
		unsigned int trans_pos = idx * rows + idy;
		mat_out[trans_pos] = mat_in[pos];
	}
}

__global__ void updatePoints(int sourceSize,int targetSize,glm::vec3 *dev_pos,glm::mat3 R, glm::vec3 trans) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= sourceSize)
		return;
	
	dev_pos[index] = R * dev_pos[index] + trans;
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void scanMatchingICP::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

void scanMatchingICP::cpuNaive(vector<glm::vec3>& source, vector<glm::vec3>& target,int iter) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
	
	
	vector<glm::vec3> sourceCorrespond, sourceNew;
	
	printf("Hello here I came in iteration:%d\n",iter);
	
	for (int i = 0; i < 5; i++) {
		printf("%0.4f %0.4f, %0.4f \n", source[i].x, source[i].y, source[i].z);
	}

	printf("For Target Points, first 5 points are: \n");

	for (int i = 0; i < 5; i++) {
		printf("%0.4f %0.4f, %0.4f \n", target[i].x, target[i].y, target[i].z);
	}

	int index;
	float dist = 0;
	float min_dist = FLT_MAX;
	for (int i = 0; i < source.size(); i++) {
		min_dist = FLT_MAX;
		for (int j = 0; j < target.size(); j++) {
			dist = glm::distance(source[i], target[j]);
			if (dist < min_dist) {
				min_dist = dist;
				index = j;
			}
		}
		sourceCorrespond.push_back(target[index]);
	}

	printf("For Correspondance Points in target, first 5 points are: \n");

	for (int i = 0; i < 10; i++) {
		printf("%0.4f %0.4f, %0.4f \n", sourceCorrespond[i].x, sourceCorrespond[i].y, sourceCorrespond[i].z);
	}

	// Mean of the traget and new Ones

	glm::vec3 meanSource(0.0f, 0.0f, 0.0f);
	glm::vec3 meanCorrespond(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < source.size(); i++) {
		meanSource += source[i];
		meanCorrespond += sourceCorrespond[i];
	}

	meanSource /= source.size();
	meanCorrespond /= source.size();

	printf("Mean of source Points: %0.4f, %0.4f, %0.4f\n", meanSource.x, meanSource.y, meanSource.z);
	printf("Mean of correspondence Points: %0.4f, %0.4f, %0.4f\n", meanCorrespond.x, meanCorrespond.y, meanCorrespond.z);

	glm::vec3 point;

	for (int i = 0; i < source.size(); i++) {
		point = source[i] - meanSource;
		sourceNew.push_back(point);
		sourceCorrespond[i] = sourceCorrespond[i] - meanCorrespond;
	}

	float W[3][3] = { 0 };
	float a, b;

	float sum = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			sum = 0;
			for (int k = 0; k < source.size(); k++) {
				a = (i == 0 ? sourceCorrespond[k].x : i == 1 ? sourceCorrespond[k].y : sourceCorrespond[k].z);
				b = (j == 0 ? sourceNew[k].x : j == 1 ? sourceNew[k].y : sourceNew[k].z);
				sum += a * b;
			}
			W[i][j] = sum;
		}
	}

	printf("The Values of Matrx Multiplication are: \n");
	
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ",W[i][j]);
		}
		printf("\n");
	}

	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
	);

	glm::mat3 g_U(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 g_Vt(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

	// Get transformation from SVD
	glm::mat3 R = g_U * g_Vt;
	glm::vec3 t = meanCorrespond - R * meanSource;

	printf("The values of U are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", g_U[i][j]);
		}
		printf("\n");
	}
	
	printf("The values of Vt are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", g_Vt[i][j]);
		}
		printf("\n");
	}

	printf("The Values of Rotation Matrix are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", R[i][j]);
		}
		printf("\n");
	}

	printf("The translational Matrix is: %0.4f, %0.4f, %0.4f\n", t.x, t.y, t.z);
	// update source points
	for (int i = 0; i < source.size(); i++) {
		source[i] = R * source[i] + t;
	}	
	//cudaMemcpy(dev_pos, &source[0], source.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pos, &source[0], source.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	//kernResetVec3Buffer << <dim3((source.size() + blockSize - 1) / blockSize), blockSize >> > (source.size(), dev_vel1, glm::vec3(1, 1, 1));
	//kernResetVec3Buffer << <dim3((target.size() + blockSize - 1) / blockSize), blockSize >> > (target.size(), &dev_vel1[source.size()], glm::vec3(1, 1, 0));

}

void scanMatchingICP::gpuNaive() {
	
	//float *W = new float[9];
	dim3 fullBlocksPerGrid((sourceSize + blockSize - 1) / blockSize);


	calculateCorrespondPoint << <fullBlocksPerGrid, blockSize >> > (sourceSize,targetSize,dev_pos,devCorrespond);
	checkCUDAErrorWithLine("Kernel CorrespondPoint failed!");
	
	
	glm::vec3 meanSource(0.0f, 0.0f, 0.0f);
	glm::vec3 meanCorrespond(0.0f, 0.0f, 0.0f);

	//thrust::device_ptr<glm::vec3> targetPtr(&dev_pos[sourceSize]);
	thrust::device_ptr<glm::vec3> sourcePtr(dev_pos);
	thrust::device_ptr<glm::vec3> correspondPtr(devCorrespond);

	meanSource = glm::vec3(thrust::reduce(sourcePtr, sourcePtr + sourceSize, glm::vec3(0, 0, 0)));
	meanCorrespond = glm::vec3(thrust::reduce(correspondPtr, correspondPtr + sourceSize, glm::vec3(0, 0, 0)));

	meanSource /= sourceSize;
	meanCorrespond /= sourceSize;

	cudaMemcpy(devTempSource,dev_pos,sourceSize * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	
	meanCentrePoints << <fullBlocksPerGrid, blockSize >> > (sourceSize,devTempSource, devCorrespond, meanSource, meanCorrespond);
	checkCUDAErrorWithLine("Kernel meanCentrePoints failed!");
	
	outerProduct << <fullBlocksPerGrid, blockSize >> > (sourceSize, devTempSource, devCorrespond,devMult);
	checkCUDAErrorWithLine("Kernel outerProduct failed!");

	glm::mat3 W = thrust::reduce(thrust::device,devMult, devMult + sourceSize, glm::mat3(0));

	//kernMatrixMultiplication << <fullBlocksPerGrid, blockSize >> > (devTempSource, devCorrespond,W,3,sourceSize,3);
	//checkCUDAErrorWithLine("Kernel Matrix Multiplication failed!");
	
	
	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][0],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
	);

	glm::mat3 g_U(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 g_Vt(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

	// Get transformation from SVD
	glm::mat3 R = g_U * g_Vt;
	glm::vec3 t = meanCorrespond - R * meanSource;

	updatePoints << <fullBlocksPerGrid, blockSize >> > (sourceSize,targetSize,dev_pos,R,t);
	checkCUDAErrorWithLine("Kernel updatePoints failed!");
	
	printf("The Values of Rotation Matrix are: \n");

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%0.4f ", R[i][j]);
		}
		printf("\n");
	}

	printf("The translational Matrix is: %0.4f, %0.4f, %0.4f\n", t.x, t.y, t.z);

	//printf("The rotation Matrix is")
	//free(check);
}

void scanMatchingICP::gpuKDTree() {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void scanMatchingICP::endSimulation() {
  cudaFree(dev_vel1);
  //cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(devCorrespond);
  cudaFree(devTempSource);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void scanMatchingICP::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
