/*
 ============================================================================
 Name        : main.cu
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */
#include <time.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "Las.h"
#include <iostream>
#include <fstream>
#include <cuComplex.h>
#include "complexExtension.h"
#include "GPUroutines.h"
#include <complex>

using namespace std;
typedef struct Qbits {
	cuDoubleComplex i[2];
} Qbit;

double bestfitness = 0;
double oldRes;

int size;
double badRes;
float * fitness;
float *tempfitness;
cuDoubleComplex *T;
cuDoubleComplex *W;
cuDoubleComplex *E;
cuDoubleComplex *A;
cuDoubleComplex *tempFitness;
cuDoubleComplex *B;
cuDoubleComplex *C;
cuDoubleComplex *Target, *Target_h;

time_t current_time;
char* c_time_string;

void printm(cuDoubleComplex *M, int nrows, int ncols, int file) {
	int i;
	if (file == 0) {
		for (i = 0; i < ncols * nrows; i++) {
			if (i % (ncols) == 0)
				printf("\n");
			printf("%lf + %lf *i ", cuCreal(M[i]), cuCimag(M[i]));
		}
	} else {
		FILE * pFile;
		pFile = fopen(c_time_string, "a");
		for (i = 0; i < ncols * nrows; i++) {
			if (i % (ncols) == 0)
				fprintf(pFile, "\n");
			fprintf(pFile, "%lf ", cuCabs(M[i]) * cuCabs(M[i]));
		}
		fprintf(pFile, "\n");
		for (i = 0; i < ncols * nrows; i++) {
			if (i % (ncols) == 0)
				fprintf(pFile, "\n");
			fprintf(pFile, "%lf + %lf *i ", cuCreal(M[i]), cuCimag(M[i]));
		}
		fprintf(pFile, "\n");
		fclose(pFile);
	}
}

double elitefitness;
Gate *population;
Gate *elite;
Gate* dev_population;
Gate* temp_population;
Gate *solution;
Gate *tempGates;
float *gpufitness;
int *parents;
cuDoubleComplex *identitym;

void fillWithZeros(cuDoubleComplex *Q, int rowsize, int colsize) {
	int i;
	for (i = 0; i < colsize * rowsize; i++) {
		Q[i] = make_cuDoubleComplex(0, 0);
	}
}
void printGates(Gate* g, int numberOfGatesToPrint) {
	int i;

	FILE * pFile;
	pFile = fopen(c_time_string, "a");
	fprintf(pFile, "\n");
	cuDoubleComplex k = make_cuDoubleComplex(-10, 0);
	for (i = 0; i < numberOfGatesToPrint; i++) {
		if (g[i].index1 == 0) {
			fprintf(pFile, "%i identity\n", i);
			fprintf(pFile, "%i %i %i %lf %lf \n", g[i].index1, g[i].index2,
					g[i].direction, g[i].parameter.x, g[i].parameter.y);

		} else if (g[i].index1 == g[i].index2) {
			fprintf(pFile, "R%i*%i(%lf)\n", g[i].index1, g[i].direction,
					g[i].parameter);

		} else if (g[i].index2 - g[i].index1 == 1
				&& comparecuDoubleComplex(g[i].parameter, k) == 1) {
			fprintf(pFile, "swap%i%i\n", g[i].index1, g[i].index2,
					g[i].parameter);
		} else {
			fprintf(pFile, "J%i%i(%lf)\n", g[i].index1, g[i].index2,
					g[i].parameter);
		}

	}
	fclose(pFile);
}
void printGatesMatlab(Gate* g, int numberOfGatesToPrint) {
	int i;

	FILE * pFile;
	pFile = fopen(c_time_string, "a");
	fprintf(pFile, "\n");
	cuDoubleComplex k = make_cuDoubleComplex(-10, 0);
	for (i = 0; i < numberOfGatesToPrint; i++) {
		fprintf(pFile, "%i %i %i %lf %lf \n", g[i].index1, g[i].index2,
				g[i].direction, g[i].parameter.x, g[i].parameter.y);
	}
	fclose(pFile);
}

void GPUGA(curandState *devStates) {
	int i, j;
	//GPU VARIABLE DECLATATION
	cuDoubleComplex *billets, *tempdata, *tempdata2, *DeviceMemory;
	cuDoubleComplex *DeviceMemory_h;
	DeviceMemory_h = new cuDoubleComplex[size * size * sizeOfPopulation];
	cublasHandle_t handle;
	cuDoubleComplex* x;
	float* gpufitness;

	double seconds;
	//GPU MEMORY ALLOCATION
	cublasCreate(&handle);
	int *solut;
	cudaMallocManaged((void **) &solut, sizeof(int));
	*solut = -1;
	cudaMalloc((void **) &(DeviceMemory),
			2 * sizeOfChromosome * sizeOfPopulation * size * size
					* sizeof(cuDoubleComplex));
	cudaMalloc((void **) &(billets),
			sizeOfChromosome * sizeOfPopulation * 4 * 4
					* sizeof(cuDoubleComplex));
	cudaMalloc((void **) &(gpufitness),
			sizeOfChromosome * sizeOfPopulation * sizeof(float));
	cudaMalloc((void **) &(tempFitness),
			sizeOfPopulation * size * size * sizeof(cuDoubleComplex));
	cudaMalloc((void **) &(tempdata),
			sizeOfChromosome * sizeOfPopulation * size * size
					* sizeof(cuDoubleComplex));
	cudaMalloc((void **) &(tempdata2),
			sizeOfChromosome * sizeOfPopulation * size * size
					* sizeof(cuDoubleComplex));
	cudaStream_t *streams = (cudaStream_t *) malloc(16 * sizeof(cudaStream_t));
	for (j = 0; j < 16; j++) {
		cudaStreamCreate(&streams[j]);
	}

	printf("In the GA\n");

	clock_t begin = clock();
	//GPURUN
	i = 0;
	int ioam;
	int iom = 0;
	srand(time(NULL));
	while (1 == 1) {
		//Construction of matrices
		i++;
		prepareBillet<<<blocksPerGrid, threadsPerBlock>>>(dev_population, billets);
		prepareMatrices<<<blocksPerGrid, threadsPerBlock>>>(dev_population, billets, tempdata, tempdata2, DeviceMemory,
				streams, size);
		for (j = 0; j < sizeOfPopulation; j++)
			cudaMemcpy(tempdata + j * size * size,
					DeviceMemory + j * sizeOfChromosome * size * size,
					size * size * sizeof(cuDoubleComplex),
					cudaMemcpyDeviceToDevice);
		//Matrix multiplication
		gpu_blas_mmul(tempdata2, tempdata, DeviceMemory, size, size, size,
				handle, size, streams);
		cudaDeviceSynchronize();
		if (sizeOfChromosome % 2 == 0)
			x = tempdata2;
		else
			x = tempdata;
		clearFitness<<<blocksPerGrid, threadsPerBlock>>>(gpufitness);
		cudaDeviceSynchronize();
		cudaMemset(tempFitness, 0, size * size * sizeof(cuDoubleComplex));
//		evaluateFitnessReversible<<<blocksPerGrid, threadsPerBlock>>>(x, Target, gpufitness, size);
		evaluateFitnessQuantum(x, Target, size, size, size, tempFitness, size,
				fitness, &handle,solut);
		cudaDeviceSynchronize();
		if (i == 0) break;
//		uncomment this if doing reversible
//		invertFitness<<<blocksPerGrid, threadsPerBlock>>>(gpufitness, solut);
		cudaDeviceSynchronize();
		if (i % checkEach == 0) {
			clock_t end = clock();
			seconds = double(end - begin) / CLOCKS_PER_SEC;
//			uncomment this if doing reversible
//			cudaMemcpy(fitness, gpufitness, sizeOfPopulation * sizeof(float), cudaMemcpyDeviceToHost);
			float aver = 0.0;
			float max = 0.0;
			iom = 0;
			for (j = 0; j < sizeOfPopulation; j++) {
				aver += fitness[j];
				if (fitness[j] > max) {
					max = fitness[j];
					iom = j;
				}
				if (verbose == 1)
					printf("%f ", fitness[j]);
			}
			aver = aver / sizeOfPopulation;
			if (max > bestfitness) {
				ioam = iom;
				bestfitness = max;

				FILE * pFile;
				pFile = fopen(c_time_string, "a");
				fprintf(pFile, "%i iteration, bestfitness is %lf\n", i,
						bestfitness);
				fprintf(pFile, "Time for %i iteration(s)  is %lf seconds \n",
						checkEach, seconds);
				fclose(pFile);
				fillWithZeros(B, size, size);
				cudaMemcpy(DeviceMemory_h, x + ioam * size * size,
						size * size * sizeof(cuDoubleComplex),
						cudaMemcpyDeviceToHost);
				printm(DeviceMemory_h, size, size, 1);

				cudaMemcpy(population, dev_population,
						sizeOfPopulation * sizeOfChromosome * sizeof(Gate),
						cudaMemcpyDeviceToHost);
				printGates(population + ioam * sizeOfChromosome,
						sizeOfChromosome);
				printGatesMatlab(population + ioam * sizeOfChromosome,
						sizeOfChromosome);
			}
			if (verbose == 1) {
				printf(
						"\nIndex of maximum is: %i, Average  fitness  is %f, Maxfitness now is: %f, Bestfitness now is: %lf \n",
						iom, aver, max, bestfitness);
				printf("iteration #%i  in %lf seconds \n", i, seconds);
			}
			begin = clock();
		}
		if (*solut != -1)
			break;
		GPUcrossover<<<blocksPerGrid, threadsPerBlock>>>(dev_population, temp_population, gpufitness, devStates, i);
		cudaDeviceSynchronize();
		mutatePopulationDev<<<blocksPerGrid, threadsPerBlock>>>(temp_population, devStates);
		cudaDeviceSynchronize();
		update_random<<<blocksPerGrid, threadsPerBlock>>>(1275, devStates);
		Gate* t;
		t = temp_population;
		temp_population = dev_population;
		dev_population = t;
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	for (j = 0; j < 16; j++) {
		cudaStreamDestroy(streams[j]);
	}
	if (*solut != -1) {
		cudaMemcpy(fitness, gpufitness, sizeOfPopulation * sizeof(float),
				cudaMemcpyDeviceToHost);
		for (j = 0; j < sizeOfPopulation; j++) {
			printf("%f ", fitness[j]);
		}

	}
	cublasDestroy(handle);
	if (*solut != -1) {
		fillWithZeros(B, size, size);
		printf("\nsolut %i\n", *solut);
		cudaMemcpy(DeviceMemory_h, x + *(solut) * size * size,
				size * size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
		printm(DeviceMemory_h, size, size, 1);
		cudaMemcpy(population, dev_population,
				sizeOfPopulation * sizeOfChromosome * sizeof(Gate),
				cudaMemcpyDeviceToHost);
		printGatesMatlab(population + *solut * sizeOfChromosome,
				sizeOfChromosome);
	}
}

int main(void) {
	int nDevices;
	FILE *pFile;
	FILE *rFile;
	current_time = time(NULL);
	c_time_string = ctime(&current_time);
	c_time_string[19] = 0;
	pFile = fopen(c_time_string, "w");
	fprintf(pFile,
			"Number of wires is:%i, mutation range %i size of Population is: %i, size of chromosome is : %i, pom is: %lf\n",
			numberOfWires, mutationRange, sizeOfPopulation, sizeOfChromosome,
			probabilityOfMutation);
	fclose(pFile);
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}
	cudaSetDevice(0);
	elite = new Gate[sizeOfChromosome];
	curandState* devStates;
	size = pow(2, numberOfWires);
	population = new Gate[sizeOfChromosome * sizeOfPopulation];
	cudaMalloc((void **) &dev_population,
			sizeOfPopulation * sizeOfChromosome * sizeof(Gate));
	cudaMalloc((void **) &temp_population,
			sizeOfPopulation * sizeOfChromosome * sizeof(Gate));
	cudaMalloc((void **) &fitness, sizeOfPopulation * sizeof(float));
	cudaMalloc((void **) &tempfitness, sizeOfPopulation * sizeof(float));
	cudaMalloc(&devStates,
			sizeOfChromosome * sizeOfPopulation * sizeof(curandState));
	setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(devStates, time(NULL));
	generatePopulationDev<<<blocksPerGrid, threadsPerBlock>>>(dev_population, devStates);
	cudaDeviceSynchronize();
	cudaMemcpy(population, dev_population,
			sizeOfPopulation * sizeOfChromosome * sizeof(Gate),
			cudaMemcpyDeviceToHost);

	solution = NULL;
	fitness = new float[sizeOfPopulation];
	parents = new int[sizeOfPopulation];
	tempGates = new Gate[sizeOfChromosome];
	int i;
	//DEFINE THE TARGET
	Target_h = new cuDoubleComplex[size * size];
	cudaMalloc(&Target, size * size * sizeof(cuDoubleComplex));
	fillWithZeros(Target_h, size, size);

	rFile = fopen("Target.txt", "r");
	if (rFile != NULL) {
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				double real;
				double imag;

				fscanf(rFile, "%lf", &real);
				fscanf(rFile, "%lf", &imag);

				Target_h[i * size + j] = make_cuDoubleComplex(real, imag);
			}
		}
		fclose(rFile);
		cudaMemcpy(Target, Target_h, size * size * sizeof(cuDoubleComplex),
				cudaMemcpyHostToDevice);
	} else {
		printf(
				"Please create a readable file Target.txt with real imaginary  number for each matrix member of a format 1.000 0.000");
		return EXIT_SUCCESS;
	}

	//PRINT THE TARGET
	printm(Target_h, size, size, 1);

	identitym = new cuDoubleComplex[size * size];
	fillWithZeros(identitym, size, size);
	for (i = 0; i < size; i++) {
		identitym[size * i + i] = make_cuDoubleComplex(1, 0);
	}

	B = new cuDoubleComplex[size * size];
	C = new cuDoubleComplex[size * size];

	//NORMAL LAUNCH

	cudaError_t cudaError;
	cudaError = cudaGetLastError();
	printf("Initialization finished with %s\n", cudaGetErrorName(cudaError));
	GPUGA(devStates);

	cudaFree(dev_population);

	delete[] fitness;
	delete[] population;

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
