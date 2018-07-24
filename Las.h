/*
 * Las.h
 *
 *  Created on: Feb 23, 2016
 *      Author: root
 */

#ifndef LAS_H_
#define LAS_H_
/*
 * transpose.h
 *
 *  Created on: 23 окт. 2015 г.
 *      Author: Georgiy
 */
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
void transpose(cuDoubleComplex * A, int nrows, int ncols);
void Kronecker_CProduct(cuDoubleComplex * C, cuDoubleComplex *A, int nrows,
                            int ncols, cuDoubleComplex *B, int mrows, int mcols);
void complexConj(cuDoubleComplex *A, int nrows, int ncols);
void multiplyC(cuDoubleComplex *C, cuDoubleComplex *A, int nrows,
                            int ncols, cuDoubleComplex *B, int mrows, int mcols);
__global__ void kronGPU(cuDoubleComplex *Result, cuDoubleComplex *First, int nrows,int ncols, cuDoubleComplex *Second, int mrows, int mcols);


#endif /* LAS_H_ */
