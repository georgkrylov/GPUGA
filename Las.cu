/*
 * transpose.c
 *
 *  Created on: 23 окт. 2015 г.
 *      Author: Georgiy
 */
#include "Las.h"
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
void transpose(cuDoubleComplex *A, int nrows, int ncols){
	int i = 0;
	int j = 0;
	cuDoubleComplex  buff;
	cuDoubleComplex  buff2;
	for (i = 0 ; i <ncols;i++){
		for (j = i+1 ; j<nrows; j++){
			buff = A[j * ncols + i];
			buff2 =A[i * ncols +j] ;
			A[j*ncols+i] = buff2;
			A[i*ncols+j]=buff;
		}
	}
}
__global__ void kronGPU(cuDoubleComplex *Result, cuDoubleComplex *First, int mrows,
	int mcols, cuDoubleComplex *Second, int nrows, int ncols)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
		while (id < ncols*nrows*mrows*mcols){
			int BlockIdx = id/(nrows*ncols);
			int ThreadIdx = id%(nrows*ncols);
			int StartOfBlock = (BlockIdx / mcols) * nrows * ncols * mcols + ncols*(BlockIdx%mcols);
			int IndexofC = StartOfBlock+ncols*mcols*(ThreadIdx/ncols)+ThreadIdx%ncols;
			int offset = IndexofC/(nrows*ncols*mrows*mcols);
			int IndexofA = offset*nrows*ncols+ThreadIdx;
			int IndexofB = offset*mrows*mcols+BlockIdx % (mrows*mcols);
			Result[IndexofC]=cuCmul(First[IndexofB],Second[IndexofA]);


		/*	printf("Block idx %i Thread idx %i StartOfBlock %i \n",BlockIdx, ThreadIdx,StartOfBlock);
			printf ("%i %i %i \n", IndexofA,IndexofB,IndexofC);*/
			id += blockDim.x * gridDim.x;
		}
}

void complexConj(cuDoubleComplex *A, int nrows, int ncols){
	int i = 0;
	int j = 0;
	cuDoubleComplex buff;
	for (i = 0 ; i <nrows;i++){
		for (j = 0 ; j<ncols; j++){
			buff = A[i*ncols + j];
			if (cuCreal(buff) == 0 && cuCimag(buff) == 0) continue;
			A[i*ncols + j] =make_cuDoubleComplex( cuCreal(buff), -cuCimag(buff) );
		}
	}
}
void Kronecker_CProduct(cuDoubleComplex *C, cuDoubleComplex *A, int nrows,
		int ncols, cuDoubleComplex *B, int mrows, int mcols)
{
	int ccols, i, j, k, l;
	int block_increment;
	cuDoubleComplex *pB;
	cuDoubleComplex *pC, *p_C;
	ccols = ncols * mcols;
	block_increment = mrows * ccols;
	for (i = 0; i < nrows; C += block_increment, i++)
		for (p_C = C, j = 0; j < ncols; p_C += mcols, A++, j++)
			for (pC = p_C, pB = B, k = 0; k < mrows; pC += ccols, k++)
				for (l = 0; l < mcols; pB++, l++) *(pC+l) =cuCmul( *A , *pB);

}


void multiplyC(cuDoubleComplex *result, cuDoubleComplex *mat1, int nrows,
		int ncols, cuDoubleComplex *mat2, int mrows, int mcols){
	int i,j,k;
	cuDoubleComplex sum;
	sum = make_cuDoubleComplex(0, 0);
	for (i = 0; i< nrows; i++){
		for (j = 0 ; j< mcols; j++){

			for (k = 0; k< ncols;k++){
				sum = cuCadd(sum,cuCmul( mat1[i*ncols+ k] , mat2[k*mcols+j]));
			}
			result[i*mcols+j] = sum;
			sum = make_cuDoubleComplex(0, 0);
		}
	}
}
