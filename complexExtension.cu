/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "complexExtension.h"

__host__ __device__ cuDoubleComplex cuCexp(cuDoubleComplex x)
{

	double real = cuCreal(x);
	double imag = cuCimag(x);
	double factor = exp(real);
	return make_cuDoubleComplex(factor * cos(imag), factor * sin(imag));
}

__host__ __device__  cuDoubleComplex cuCcos(cuDoubleComplex x){
	double real = cuCreal(x);
	double imag = cuCimag(x);
	return make_cuDoubleComplex(cos(real)*cosh(imag), -sin(real)*sinh(imag));

}
__host__ __device__ int comparecuDoubleComplex(cuDoubleComplex a, cuDoubleComplex  b){
	if (cuCimag (a) == cuCimag(b) && cuCreal(a) == cuCreal(b) ) return 1;
	return 0;
}

__host__ __device__ cuDoubleComplex cuCsin(cuDoubleComplex x){
	double real = cuCreal(x);
	double imag = cuCimag(x);

	return make_cuDoubleComplex(sin(real)*cosh(imag), cos(real)*sinh(imag));

}
