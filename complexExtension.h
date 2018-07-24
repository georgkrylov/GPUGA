/*
 * complexExtension.h
 *
 *  Created on: Mar 16, 2016
 *      Author: root
 */

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


__host__ __device__ cuDoubleComplex cuCsin(cuDoubleComplex x);
__host__ __device__ cuDoubleComplex cuCcos(cuDoubleComplex x);
__host__ __device__ int comparecuDoubleComplex(cuDoubleComplex a, cuDoubleComplex  b);
__host__ __device__  cuDoubleComplex cuCexp(cuDoubleComplex x);



