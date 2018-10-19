#include "GPUroutines.h"
__global__ void setup_kernel(curandState * state, unsigned long seed) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	while (id < sizeOfChromosome * sizeOfPopulation) {
		curand_init(seed, id, 0, &state[id]);
		id += blockDim.x * gridDim.x;

	}
}
__global__ void update_random(unsigned long long n, curandState_t *state) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	while (id < sizeOfChromosome * sizeOfPopulation) {
		skipahead(n, &(state[id]));
		id += blockDim.x * gridDim.x;
	}
}

__global__ void GPUcrossover(Gate * population, Gate *temp_population,
		float *fitness, curandState *globalState, int wer) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state;

	while (tid < sizeOfPopulation) {
		state = globalState[tid];
		int i, j;
		float a[window];
		int b[window];
		for (i = 0; i < window; i++) {
			a[i] = curand_uniform(&state);
			b[i] = (int) ((curand_uniform(&state)) * (float) sizeOfPopulation);
		}

		b[0] = tid;
		for (i = 0; i < window; i++) {
			for (j = 0; j < window; j++) {
				if (fitness[b[i]] > fitness[b[j]]) {
					a[i] += 1.0;
				}
			}
		}

		int iof = b[0];
		int ios = b[0];

		float m1 = a[0];
		for (i = 0; i < window; i++) {
			if (m1 < a[i]) {
				iof = b[i];
				m1 = a[i];
			}
		}
		m1 = -1;
		for (i = 0; i < window; i++) {
			if (m1 <= a[i] && b[i] != iof) {
				ios = b[i];
				m1 = a[i];
			}
		}
		if (ios == -1)
			ios = iof;

		int p1 = (int) ((curand_uniform_double(&state)) * sizeOfChromosome);
		int p2 = (int) ((curand_uniform_double(&state)) * sizeOfChromosome);
		double r = curand_uniform_double(&state);

		if (r > probabilityOfCrossover) {
			iof = tid;
			ios = tid;
		}

		if (p1 > p2) {
			p2 += p1;
			p1 = p2 - p1;
			p2 = p2 - p1;
		}
		for (i = 0; i < p1; i++) {
			temp_population[tid * sizeOfChromosome + i] = population[iof
					* sizeOfChromosome + i];

		}
		for (i = p1; i < p2; i++) {
			temp_population[tid * sizeOfChromosome + i] = population[ios
					* sizeOfChromosome + i];

		}
		for (i = p2; i < sizeOfChromosome; i++) {
			temp_population[tid * sizeOfChromosome + i] = population[iof
					* sizeOfChromosome + i];

		}

		tid += blockDim.x * gridDim.x;
	}
}

// CUDA Kernel for Vector Addition
__global__ void generatePopulationDev(Gate * g, curandState *globalState) {
	//Get the id of thread within a block
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < sizeOfChromosome * sizeOfPopulation) {
		curandState state = globalState[tid];
		int i = tid;
		int r;
		double parameter;
		int k = 1
				+ ((int) ((curand_uniform_double(&state) * scaleForRandom)))
						% (mutationRange);
		//k*=2;
		parameter = PI / k;
		int sign = ((int) ((curand_uniform_double(&state) * scaleForRandom)))
				% 2;
		if (sign == 0)
			parameter = 1 * parameter;
		else
			parameter = -parameter;
		k = 1 + (((int) (curand_uniform_double(&state) * scaleForRandom)) % k);
		//k=1;
		parameter *= k;
		g[i].parameter = make_cuDoubleComplex(parameter, 0);
		if (numberOfWires > 1) {
			if (numberOfWires > 2)
				r =
						1
								+ ((int) (curand_uniform_double(&state)
										* scaleForRandom)) % 8;
			else
				r =
						1
								+ ((int) (curand_uniform_double(&state)
										* scaleForRandom)) % 3;
			if (r == 1 || r == 3 || r == 5 || r == 4 || r == 6 || r == 7) {
				g[i].index1 =
						1
								+ ((int) (curand_uniform_double(&state)
										* scaleForRandom)) % numberOfWires;
				g[i].index2 = g[i].index1;
				g[i].direction = ((int) (curand_uniform_double(&state)
						* scaleForRandom)) % 3;
			}
			if (r == 2) {
				int ri1;
				if (numberOfWires == 2) {
					ri1 = 1;
				} else {
					ri1 = 1
							+ ((int) (curand_uniform_double(&state)
									* scaleForRandom)) % (numberOfWires - 1);
				}
				int ri2 = ri1 + 1;
				g[i].index1 = ri1;
				g[i].index2 = ri2;
				g[i].direction = ((int) (curand_uniform_double(&state)
						* scaleForRandom)) % 3;
			}

			if (r == 8) {
				int ri1;
				if (numberOfWires == 2) {
					ri1 = 1;
				} else {
					ri1 = 1
							+ ((int) (curand_uniform_double(&state)
									* scaleForRandom)) % (numberOfWires - 1);
				}
				int ri2 = ri1 + 1;
				g[i].index1 = ri1;
				g[i].index2 = ri2;
				g[i].parameter = make_cuDoubleComplex(-10, 0);
				g[i].direction = ((int) (curand_uniform_double(&state)
						* scaleForRandom)) % 3;
			}
		} else {
			r = 1 + ((int) curand_uniform_double(&state)) % 2;
			if (r == 1) {
				g[i].index1 =
						1
								+ ((int) (curand_uniform_double(&state)
										* scaleForRandom)) % numberOfWires;
				g[i].index2 = g[i].index1;
				g[i].direction = ((int) (curand_uniform_double(&state)
						* scaleForRandom)) % 3;
			}
			if (r == 2) {
				g[i].index1 = 0;
				g[i].index2 = 0;
			}
		}
		tid += blockDim.x * gridDim.x;
	}

}
__global__ void mutatePopulationDev(Gate * g, curandState *globalState) {
	//Get the id of thread within a block
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < sizeOfChromosome * sizeOfPopulation) {
		curandState state = globalState[tid];
		int i = tid;
		int r;
		double mutationp = curand_uniform_double(&state);

		if (mutationp < probabilityOfMutation) {

			mutationp = curand_uniform_double(&state);
			if (mutationp < 0.5) {

				if (numberOfWires >= 2) {
					if (numberOfWires > 2)
						r = 1
								+ ((int) (curand_uniform_double(&state)
										* scaleForRandom)) % 3;
					else
						r = 1
								+ ((int) (curand_uniform_double(&state)
										* scaleForRandom)) % 2;
					double parameter;
					int k = 1
							+ ((int) ((curand_uniform_double(&state)
									* scaleForRandom))) % (mutationRange);
					parameter = PI / k;
					int sign = ((int) (curand_uniform_double(&state)
							* scaleForRandom)) % 2;
					if (sign == 0)
						parameter = 1 * parameter;
					else
						parameter = -parameter;
					k = 1
							+ (((int) (curand_uniform_double(&state)
									* scaleForRandom)) % k);
					parameter *= k;
					g[i].parameter = make_cuDoubleComplex(parameter, 0);

					if (r == 1) {
						g[i].index1 = 1
								+ ((int) (curand_uniform_double(&state)
										* scaleForRandom)) % numberOfWires;
						g[i].index2 = g[i].index1;
						g[i].direction = ((int) (curand_uniform_double(&state)
								* scaleForRandom)) % 3;
					}

					if (r == 2) {
						int ri1;
						if (numberOfWires == 2) {
							ri1 = 1;
						} else {
							ri1 = 1
									+ ((int) (curand_uniform_double(&state)
											* scaleForRandom))
											% (numberOfWires - 1);
						}
						int ri2 = ri1 + 1;
						g[i].index1 = ri1;
						g[i].index2 = ri2;
					}

					if (r == 3) {
						int ri1;
						if (numberOfWires == 2) {
							ri1 = 1;
						} else {
							ri1 = 1
									+ ((int) (curand_uniform_double(&state)
											* scaleForRandom))
											% (numberOfWires - 1);
						}
						int ri2 = ri1 + 1;
						g[i].index1 = ri1;
						g[i].index2 = ri2;
						g[i].parameter = make_cuDoubleComplex(-10, 0);
					}
				}

				else {
					r = 1
							+ ((int) (curand_uniform_double(&state)
									* scaleForRandom)) % 2;
					if (r == 1) {
						g[i].index1 = 1
								+ ((int) (curand_uniform_double(&state)
										* scaleForRandom)) % numberOfWires;
						g[i].index2 = g[i].index1;
						g[i].direction = ((int) (curand_uniform_double(&state)
								* scaleForRandom)) % 3;
					}
				}
			}

			else if (mutationp > 0.5) {

				cuDoubleComplex k = make_cuDoubleComplex(-10, 0);
				if (comparecuDoubleComplex(g[i].parameter, k) != 1) {
					double parameter = cuCreal(g[i].parameter);
					double delta = PI / (mutationRange);
					r = ((int) (curand_uniform_double(&state) * scaleForRandom))
							% 2;
					if (r < 1) {
						//	printf("Here\n");
						parameter += delta;
						if (parameter > PI)
							parameter -= 2 * PI;
						cuDoubleComplex temp = make_cuDoubleComplex(parameter,
								0);
						g[i].parameter = temp;

					} else {
						//	printf("Zere\n");
						parameter -= delta;
						if (parameter < -PI)
							parameter += 2 * PI;
						cuDoubleComplex temp = make_cuDoubleComplex(parameter,
								0);
						g[i].parameter = temp;
					}

				}
			}
		}
		tid += blockDim.x * gridDim.x;
	}

}
__global__ void prepareBillet(Gate *G, cuDoubleComplex * memory) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	cuDoubleComplex k = make_cuDoubleComplex(-10, 0);
	while (tid < sizeOfChromosome * sizeOfPopulation) {
		Gate g = G[tid];
		int i;
		for (i = 0; i < 4 * 4; i++) {
			memory[tid * 4 * 4 + i] = make_cuDoubleComplex(0, 0);
		}

		//same indexes mean this is an rotational gate
		if (g.index1 == g.index2) {
			switch (g.direction) {
			case X:

				memory[tid * 4 * 4 + 0] = cuCcos(
						cuCdiv(g.parameter, make_cuDoubleComplex(2, 0)));
				memory[tid * 4 * 4 + 1] = cuCmul(make_cuDoubleComplex(0, -1),
						cuCsin(
								cuCdiv(g.parameter,
										make_cuDoubleComplex(2, 0))));
				memory[tid * 4 * 4 + 2] = cuCmul(make_cuDoubleComplex(0, -1),
						cuCsin(
								cuCdiv(g.parameter,
										make_cuDoubleComplex(2, 0))));
				memory[tid * 4 * 4 + 3] = cuCcos(
						cuCdiv(g.parameter, make_cuDoubleComplex(2, 0)));
				break;
			case Y:

				memory[tid * 4 * 4 + 0] = cuCcos(
						cuCdiv(g.parameter, make_cuDoubleComplex(2, 0)));
				memory[tid * 4 * 4 + 1] = cuCsin(
						cuCdiv(g.parameter, make_cuDoubleComplex(-2, 0)));
				memory[tid * 4 * 4 + 2] = cuCsin(
						cuCdiv(g.parameter, make_cuDoubleComplex(2, 0)));
				memory[tid * 4 * 4 + 3] = cuCcos(
						cuCdiv(g.parameter, make_cuDoubleComplex(2, 0)));
				break;

			case Z:

				memory[tid * 4 * 4 + 0] = cuCexp(
						cuCmul(make_cuDoubleComplex(0, -1),
								cuCdiv(g.parameter,
										make_cuDoubleComplex(2, 0))));
				memory[tid * 4 * 4 + 1] = make_cuDoubleComplex(0, 0);
				memory[tid * 4 * 4 + 2] = make_cuDoubleComplex(0, 0);
				memory[tid * 4 * 4 + 3] = cuCexp(
						cuCmul(make_cuDoubleComplex(0, 1),
								cuCdiv(g.parameter,
										make_cuDoubleComplex(2, 0))));
				break;
			}
		}
		//indexes differing by one mean it is interaction gate
		if (abs(g.index1 - g.index2) == 1
				&& comparecuDoubleComplex(g.parameter, k) == 0) {
			memory[tid * 4 * 4 + 0] = make_cuDoubleComplex(1, 0);
			memory[tid * 4 * 4 + 5] = cuCexp(
					cuCmul(g.parameter, make_cuDoubleComplex(0, 1)));
			memory[tid * 4 * 4 + 10] = cuCexp(
					cuCmul(g.parameter, make_cuDoubleComplex(0, 1)));
			memory[tid * 4 * 4 + 15] = make_cuDoubleComplex(1, 0);
			for (i = 0; i < 4 * 4; i++) {
				memory[tid * 4 * 4 + i] = cuCmul(memory[tid * 4 * 4 + i],
						cuCexp(
								cuCmul(make_cuDoubleComplex(0, -1),
										cuCdiv(g.parameter,
												make_cuDoubleComplex(2, 0)))));
			}
		}
		//indexes differing more than by one mean there should be swap gates
		if (abs(g.index1 - g.index2) == 1
				&& comparecuDoubleComplex(g.parameter, k) == 1) {
			memory[tid * 4 * 4 + 0] = make_cuDoubleComplex(1, 0);
			memory[tid * 4 * 4 + 6] = make_cuDoubleComplex(1, 0);
			memory[tid * 4 * 4 + 9] = make_cuDoubleComplex(1, 0);
			memory[tid * 4 * 4 + 15] = make_cuDoubleComplex(1, 0);

		}

		tid += blockDim.x * gridDim.x;

	}
}
__global__ void prepareMatrices(Gate *G, cuDoubleComplex * sourcememory,
		cuDoubleComplex * tempmemory, cuDoubleComplex * tempmemory2,
		cuDoubleComplex * resultmemory, cudaStream_t * streams, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	cuDoubleComplex k = make_cuDoubleComplex(-10, 0);
	while (tid < sizeOfChromosome * sizeOfPopulation) {

		Gate g = G[tid];
		int i;
		int size1 = (int) pow((double) 2, (double) (g.index1 - 1));

		for (i = 0; i < size1 * size1; i++) {
			tempmemory[tid * size * size + i] = make_cuDoubleComplex(0, 0);
		}
		for (i = 0; i < size1; i++) {

			tempmemory[tid * size * size + i * (size1) + i] =
					make_cuDoubleComplex(1, 0);
		}
		if (g.index1 == g.index2) {
			kronGPU<<<4, 32>>>(tempmemory2 + tid * size * size, tempmemory + tid * size * size, size1, size1,
					sourcememory + tid * 4 * 4, 2, 2);
		}
		if (abs(g.index1 - g.index2) == 1
				&& comparecuDoubleComplex(g.parameter, k) == 0) {
			kronGPU<<<4, 32>>>(tempmemory2 + tid * size * size, tempmemory + tid * size * size, size1, size1,
					sourcememory + tid * 4 * 4, 4, 4);
		}
		if (abs(g.index1 - g.index2) == 1
				&& comparecuDoubleComplex(g.parameter, k) == 1) {
			kronGPU<<<4, 32>>>(tempmemory2 + tid * size * size, tempmemory + tid * size * size, size1, size1,
					sourcememory + tid * 4 * 4, 4, 4);
		}
		int size2 = (int) pow((double) 2, (double) (numberOfWires - g.index2));
		cudaDeviceSynchronize();
		for (i = 0; i < size2 * size2; i++) {
			tempmemory[tid * size * size + i] = make_cuDoubleComplex(0, 0);
		}
		for (i = 0; i < size2; i++) {
			tempmemory[tid * size * size + i * (size2) + i] =
					make_cuDoubleComplex(1, 0);
		}
		if (g.index1 == g.index2) {
			kronGPU<<<4, 32>>>(resultmemory + tid * size * size, tempmemory2 + tid * size * size, size1 * 2, size1 * 2,
					tempmemory + tid * size * size, size2, size2);
		}
		if (abs(g.index1 - g.index2) == 1
				&& comparecuDoubleComplex(g.parameter, k) == 0) {
			kronGPU<<<4, 32>>>(resultmemory + tid * size * size, tempmemory2 + tid * size * size, size1 * 4, size1 * 4,
					tempmemory + tid * size * size, size2, size2);
		}
		if (abs(g.index1 - g.index2) == 1
				&& comparecuDoubleComplex(g.parameter, k) == 1) {
			kronGPU<<<4, 32>>>(resultmemory + tid * size * size, tempmemory2 + tid * size * size, size1 * 4, size1 * 4,
					tempmemory + tid * size * size, size2, size2);
		}

		tid += blockDim.x * gridDim.x;
	}
}
void gpu_blas_mmul(cuDoubleComplex* temp_circuit,
		cuDoubleComplex* final_circuit, cuDoubleComplex* Gates, const int m,
		const int k, const int n, cublasHandle_t handle, int size,
		cudaStream_t * streams) {

	int lda = m, ldb = k, ldc = m;
	int i, j;

	const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
	const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
	const cuDoubleComplex *alpha = &alf;
	const cuDoubleComplex *beta = &bet;
	//cudaStream_t* stream;
	cuDoubleComplex *t_t;
	for (i = 1; i < sizeOfChromosome; i++) {
		// Launch each DGEMM operation in own CUDA stream
		for (j = 0; j < sizeOfPopulation; j++) {
			// Set CUDA stream

			//	cublasSetStream(handle, streams[j%16]);
			//	printf("%i %i lala %i\n",i,j,j*sizeOfChromosome+i);
			// ZGEMM: C = alpha*A*B + beta*C
			cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
					Gates + size * size * j * sizeOfChromosome
							+ i * size * size, lda,
					final_circuit + size * size * j, ldb, beta,
					temp_circuit + size * size * j, ldc);
			//	cudaDeviceSynchronize();

		}
		/*	for (j = 0 ; j<16;j++){
		 cublasGetStream(handle, stream);
		 cudaStreamSynchronize(*stream);
		 }*/
		cudaDeviceSynchronize();

		t_t = final_circuit;
		final_circuit = temp_circuit;
		temp_circuit = t_t;
	}

}

__global__ void evaluateFitnessReversible(cuDoubleComplex * Data,
		cuDoubleComplex * Target, float *tempfitness, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < sizeOfPopulation * size * size) {
		atomicAdd(tempfitness + tid / size / size,
				(float) ((cuCabs(Data[tid])
						- cuCabs(Target[tid % (size * size)]))
						* (cuCabs(Data[tid])
								- cuCabs(Target[tid % (size * size)]))));
		tid += blockDim.x * gridDim.x;
	}
}
void evaluateFitnessQuantum(cuDoubleComplex * Data, cuDoubleComplex * Target,
		const int m, const int k, const int n, cuDoubleComplex* tempfitness,
		int size, float* fitness, cublasHandle_t* h,int* solut) {
	cublasHandle_t handle = (cublasHandle_t) *h;
	int lda = m, ldb = k, ldc = m;
	const cuDoubleComplex alf = make_cuDoubleComplex(1, 0);
	const cuDoubleComplex bet = make_cuDoubleComplex(0, 0);
	const cuDoubleComplex *alpha = &alf;
	const cuDoubleComplex *beta = &bet;
	int i;
	for (i = 0; i < sizeOfPopulation; i++) {
		cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, alpha, Target,
				lda, Data + size * size * i, ldb, beta,
				tempfitness + size * size * i, ldc);
	}
	for (i = 0 ; i < sizeOfPopulation; i++){
		printf("Individual number %i\n", i);
		for (int j = 0 ; j < size; j++){
			for (int k = 0; k < size; k++){
				printf("%lf + %lf + j ", cuCreal(tempfitness[size*size*i+j*size+k]),cuCimag(tempfitness[size*size*i+j*size+k]));
			}
			printf("\n");
		}
	}
	cudaDeviceSynchronize();
	int dSize = size * size * sizeOfPopulation;
	cuDoubleComplex hData[dSize];
	cudaMemcpy(hData, tempfitness, dSize * sizeof(cuDoubleComplex),
			cudaMemcpyDeviceToHost);
	for (i = 0; i < sizeOfPopulation; i++) {
		cuDoubleComplex tSum = make_cuDoubleComplex(0, 0);
		int j;
		int matrixOffset = i * size * size;
		for (j = 0; j < size * size; j += (size + 1)) {
			tSum = cuCadd(tSum, *(hData + matrixOffset + j));
		}
		fitness[i] = cuCabs(tSum);
	}
	for (i = 0; i < sizeOfPopulation; i++) {
		fitness[i] = 1-(float) sqrt((size * 1.0 - fitness[i]) / (size));
		if (fitness[i] == 1)
			*solut = i;
	}
}
__global__ void clearFitness(float *tempfitness) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < sizeOfPopulation) {
		tempfitness[tid] = 0;
		tid += blockDim.x * gridDim.x;
	}

}
__global__ void invertFitness(float *tempfitness, int* solut) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < sizeOfPopulation) {
		float a = 1 + tempfitness[tid];
		if (1 / a >= 1 - tol)
			*solut = tid;
		tempfitness[tid] = 1.0 / a;
		tid += blockDim.x * gridDim.x;
	}

}
