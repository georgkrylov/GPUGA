#ifndef GPUROUTINES_H_
#define GPUROUTINES_H_
#include <curand_kernel.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include "parameters.h"
#include "Las.h"
#include "complexExtension.h"

__global__ void setup_kernel(curandState * state, unsigned long seed);
// CUDA Kernel for Vector Addition
__global__ void generatePopulationDev(Gate * g, curandState *globalState);
__global__ void prepareBillet(Gate *G, cuDoubleComplex * memory);
__global__ void prepareMatrices(Gate *G, cuDoubleComplex * sourcememory, cuDoubleComplex * tempmemory,
		cuDoubleComplex * tempmemory2, cuDoubleComplex * resultmemory, cudaStream_t * streams, int size);
void gpu_blas_mmul(cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, const int, const int, const int,
		cublasHandle_t, int, cudaStream_t*);
__global__ void evaluateFitnessGPU(cuDoubleComplex *, cuDoubleComplex *, float*, int);
__global__ void clearFitness(float *);
__global__ void mutatePopulationDev(Gate *, curandState*);
__global__ void invertFitness(float *, int*);
__global__ void test_random(curandState * globalState);
__global__ void update_random(unsigned long long n, curandState_t *state);
__global__ void GPUcrossover(Gate * population, Gate *temp_population, float *fitness, curandState *globalState,
		int wer);
#endif /* GPUROUTINES_H_ */
