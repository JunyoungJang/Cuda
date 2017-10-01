
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#define PI 3.141592653589793
#define NumThread 64
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CHECK_TIME_START __int64 freq, start, end; if (QueryPerformanceFrequency((_LARGE_INTEGER*)&freq)) {QueryPerformanceCounter((_LARGE_INTEGER*)&start);
#define CHECK_TIME_END(a,b) QueryPerformanceCounter((_LARGE_INTEGER*)&end); a=(float)((double)(end - start)/freq*1000); b=TRUE; } else b=FALSE;
// Check err for GPU.
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// To use GPU
__global__
void poissonmatrixKernel(int firstrowindex, int firstgrid_x, int lastgrid_x, int numgrid_x, int numgrid_y, float* poissonmatrix) {
	int count, ii, jj, kk, cur_rindx, lindx;
	int columnsize = numgrid_x*numgrid_y;

	count = 0;

	for (ii = firstgrid_x; ii <= lastgrid_x; ii++) {
		for (jj = 1; jj <= numgrid_y; jj++) {
			cur_rindx = (ii - 1)*numgrid_y + jj - 1;
			lindx = cur_rindx - firstrowindex + 1;
			if (ii == 1) {
				if (jj == 1) {
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
				else if (jj == numgrid_y) {
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
				else {
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
			}
			else if (ii == numgrid_x) {
				if (jj == 1) {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
				}
				else if (jj == numgrid_y) {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
				}
				else {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
				}
			}
			else {
				if (jj == 1) {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}

				else if (jj == numgrid_y) {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
				else {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
			}
		}
	}
}

__device__
void innerproductKernel(float *vec1, float *vec2, int size, float *result) {
	result[0] = 0.0;
	float temp = 0.0;

	for (int i = 0; i < size; i++)
		temp += vec1[i] * vec2[i];
	atomicAdd(&result[0], temp);
}

__device__
void multiplyKernel(float *matrix, float *vec, int size, int VecSize, float *result) {
	float sum = 0;


	for (int i = 0; i < size; i++){
		sum = 0.0;
		for (int j = 1; j< (size); j++)
			sum += matrix[i*size + j] * vec[j];
		atomicAdd(&result[i], sum);
	}
	
}

__global__
void cgsolver(int size, int VecSize, float *matrix, float *rhs, float *solution0, float *solution1,
	float *res0tol, float *temp1, float *temp2, float *alpha , float *beta, 
	float *res0, float *res1, float *p0, float *p1, float *Ax, float *Ap,
	int maxiteration, float tolerance) {

	int offset = threadIdx.x * size + blockIdx.x * blockDim.x;// blockIdx.x * VecSize + threadIdx.x;
	float sum[1];

	if (offset + size <= size) {
		// MATRIX Multiply
		for (int i = 0; i < size; i++) {
			sum[0] = 0.0;
			for (int j = offset; j < (offset + size); j++)
				sum[0] += matrix[i*size + j] * solution0[j];
			atomicAdd(&Ax[i], sum[0]);}

		for (int i = offset; i < (offset + size); i++) {
			res1[i] = rhs[i] - Ax[i];
			p1[i] = res1[i];}

		innerproductKernel(res1, res1, size, res0tol);
		for (int iter = 0; iter<maxiteration; iter++) {
			//if ((iter % 20 == 0) && (iter != 0) && (offset == 0))
				//printf("[CG] mse %e with a tolerance criteria of %e at %5d iterations.\n", sqrt(temp2[0] / res0tol[0]), tolerance, iter);
			
			for (int i = offset; i < (offset + size); i++) {
				res0[i] = res1[i];
				p0[i] = p1[i];
				solution0[i] = solution1[i];}

			innerproductKernel(res0, res0, size, temp1);
		
			for (int i = 0; i < size; i++) {
				sum[0] = 0.0;
				for (int j = offset; j < (offset + size); j++)
					sum[0] += matrix[i*size + j] * p0[j];
				atomicAdd(&Ap[i], sum[0]);}

			innerproductKernel(Ap, p0, size, temp2);
			
			alpha[0] = temp1[0] / temp2[0];
			
			for (int i = offset; i < (offset + size); i++) {
				solution1[i] = solution0[i] + alpha[0] * p0[i];
				res1[i] = res0[i] - alpha[0] * Ap[i];}
			
			innerproductKernel(res1, res1, size, temp2);
			
			if (sqrt(temp2[0] / res0tol[0]) < tolerance)
				break;

			beta[0] = temp2[0] / temp1[0];
			
			for (int i = offset; i < (offset + size); i++) {
				p1[i] = res1[i] + beta[0] * p0[i];
				solution0[i] = 0.0;
				Ap[i] = 0.0;
				res0[i] = 0.0;
				p0[i] = 0.0;
			}
			alpha[0] = 0.0;
			beta[0] = 0.0;
		}

	}
}

__global__
void CRScgsolver(int size, int VecSize, int CRS_size, float *nnz, int *col_ind, int *row_ind, float *rhs, float *solution0, float *solution1,
	float *res0tol, float *temp1, float *temp2, float *alpha, float *beta,
	float *res0, float *res1, float *p0, float *p1, float *Ax, float *Ap,
	int maxiteration, float tolerance) {

	int offset = threadIdx.x * size + blockIdx.x * blockDim.x;// blockIdx.x * VecSize + threadIdx.x;
	float sum[1];

	if (offset + size <= size) {
		// MATRIX Multiply
		for (int i = 0; i < size; i++) {
			sum[0] = 0.0;
			for (int j = offset; j < (offset + size); j++)
				for (int k = row_ind[j];k <= (row_ind[i+1]-1);k++)
					sum[0] += nnz[k] * solution0[col_ind[k]];
			atomicAdd(&Ax[i], sum[0]);
		}

		for (int i = offset; i < (offset + size); i++) {
			res1[i] = rhs[i] - Ax[i];
			p1[i] = res1[i];
		}

		innerproductKernel(res1, res1, size, res0tol);
		for (int iter = 0; iter<maxiteration; iter++) {
			//if ((iter % 20 == 0) && (iter != 0) && (offset == 0))
			//printf("[CG] mse %e with a tolerance criteria of %e at %5d iterations.\n", sqrt(temp2[0] / res0tol[0]), tolerance, iter);

			for (int i = offset; i < (offset + size); i++) {
				res0[i] = res1[i];
				p0[i] = p1[i];
				solution0[i] = solution1[i];
			}

			innerproductKernel(res0, res0, size, temp1);

			for (int i = 0; i < size; i++) {
				sum[0] = 0.0;
				for (int j = offset; j < (offset + size); j++)
					for (int k = row_ind[j]; k <= (row_ind[i + 1] - 1); k++)
						sum[0] += nnz[k] * p0[col_ind[k]];
				atomicAdd(&Ap[i], sum[0]);
			}

			innerproductKernel(Ap, p0, size, temp2);

			alpha[0] = temp1[0] / temp2[0];

			for (int i = offset; i < (offset + size); i++) {
				solution1[i] = solution0[i] + alpha[0] * p0[i];
				res1[i] = res0[i] - alpha[0] * Ap[i];
			}

			innerproductKernel(res1, res1, size, temp2);

			if (sqrt(temp2[0] / res0tol[0]) < tolerance)
				break;

			beta[0] = temp2[0] / temp1[0];

			for (int i = offset; i < (offset + size); i++) {
				p1[i] = res1[i] + beta[0] * p0[i];
				solution0[i] = 0.0;
				Ap[i] = 0.0;
				res0[i] = 0.0;
				p0[i] = 0.0;
			}
			alpha[0] = 0.0;
			beta[0] = 0.0;
		}

	}
}




// To use CPU
void construct_poissonmatrix(int firstrowindex, int firstgrid_x, int lastgrid_x, int numgrid_x, int numgrid_y, float *poissonmatrix)
{
	int count, ii, jj, kk, cur_rindx, lindx;
	int columnsize = numgrid_x*numgrid_y;

	count = 0;

	for (ii = firstgrid_x; ii <= lastgrid_x; ii++) {
		for (jj = 1; jj <= numgrid_y; jj++) {

			cur_rindx = (ii - 1)*numgrid_y + jj - 1;
			lindx = cur_rindx - firstrowindex + 1;
			if (ii == 1) {
				if (jj == 1) {
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
				else if (jj == numgrid_y) {
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
				else {
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
			}
			else if (ii == numgrid_x) {
				if (jj == 1) {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
				}
				else if (jj == numgrid_y) {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
				}
				else {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
				}
			}
			else {
				if (jj == 1) {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}

				else if (jj == numgrid_y) {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
				else {
					poissonmatrix[lindx*columnsize + cur_rindx - numgrid_y] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx - 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx] = -4.0;
					poissonmatrix[lindx*columnsize + cur_rindx + 1] = 1.0;
					poissonmatrix[lindx*columnsize + cur_rindx + numgrid_y] = 1.0;
				}
			}
		}
	}
}






void cgsolver(int size, float *matrix, float *rhs, float *solution, int maxiteration, float tolerance) {
	float *res0, *p0, *Ax, *Ap, *res1, *p1;
	float *MatKernel, *rhsKernel, *solutionKernel0, *solutionKernel1, *res0tol, *temp1, *temp2, *alpha, *beta;
	int VecSizeKernel = size * sizeof(float);
	int MatSizeKernel = size * size * sizeof(float);
	int SolutionSize = (int)sqrt(size);

	cudaMalloc((void**)&res0tol, sizeof(float));
	cudaMalloc((void**)&temp1, sizeof(float));
	cudaMalloc((void**)&temp2, sizeof(float));
	cudaMalloc((void**)&alpha, sizeof(float));
	cudaMalloc((void**)&beta, sizeof(float));
	cudaMalloc((void**)&res0, VecSizeKernel);
	cudaMalloc((void**)&p0, VecSizeKernel);
	cudaMalloc((void**)&Ax, VecSizeKernel);
	cudaMalloc((void**)&Ap, VecSizeKernel);
	cudaMalloc((void**)&res1, VecSizeKernel);
	cudaMalloc((void**)&p1, VecSizeKernel);

	cudaMalloc((void**)&MatKernel, MatSizeKernel);
	cudaMemcpy(MatKernel, matrix, MatSizeKernel, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&rhsKernel, VecSizeKernel);
	cudaMemcpy(rhsKernel, rhs, VecSizeKernel, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&solutionKernel0, VecSizeKernel);
	cudaMemcpy(solutionKernel0, solution, VecSizeKernel, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&solutionKernel1, VecSizeKernel);
	cudaMemcpy(solutionKernel1, solution, VecSizeKernel, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float time;
	
	printf("NumOfThread\tTime for the kernel");
	int NumOfThread = 1;
	for (int i = 0; i < 1; i++) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		cgsolver << <(VecSizeKernel + NumOfThread - 1) / NumOfThread, NumOfThread >> >(size, SolutionSize, MatKernel,
			rhsKernel, solutionKernel0, solutionKernel1, res0tol, temp1, temp2, alpha, beta,
			res0, p0, Ax, Ap, res1, p1, maxiteration, tolerance);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		
		printf("%f\n", time/1000);
		NumOfThread = NumOfThread + 1;
	}
	
	



	cudaMemcpy(solution, solutionKernel1, VecSizeKernel, cudaMemcpyDeviceToHost);

	cudaFree(res0);
	cudaFree(p0);
	cudaFree(res1);
	cudaFree(p1);
	cudaFree(Ax);
	cudaFree(Ap);
	cudaFree(MatKernel);
	cudaFree(rhsKernel);
	cudaFree(solutionKernel0);
	cudaFree(solutionKernel1);
}


void CRS_pmatrix(int matrix_size, int CRS_iter, float *nnz, int *row_ptr, int *col_ind) {
	
}




void CRS_cgsolver(int size, float *rhs, float *solution, int maxiteration, float tolerance) {
	float *res0, *p0, *Ax, *Ap, *res1, *p1;
	float *MatKernel, *rhsKernel, *solutionKernel0, *solutionKernel1, *res0tol, *temp1, *temp2, *alpha, *beta;
	int SolutionSize = (int)sqrt(size);

	float *nnz, *nnz1, *nnzKernel;
	int *row_ptr, *row_ptrKernel, *col_ind, *col_ind1, *col_indKernel, CRS_iter;
	nnz = (float*)malloc(sizeof(float)*size*size * 5);
	row_ptr = (int*)malloc(sizeof(int)*size*size);
	col_ind = (int*)malloc(sizeof(int)*size*size * 5);
	
	CRS_iter = 3;
	for (int i = 0; i < size*size * 5; i++) {
		nnz[i] = 0.0;
		col_ind[i] = 0;
	}
	for (int i = 0; i < size*size; i++)
		row_ptr[i] = 0;

	CRS_iter = 3;
	int i, j;
	int  NumOfGrid = size;
	nnz[0] = -4.0;
	nnz[1] = 2.0;
	nnz[2] = 1.0;
	row_ptr[0] = 0;
	col_ind[0] = 0;
	col_ind[1] = 1;
	col_ind[2] = NumOfGrid;

	for (i = 2; i <= NumOfGrid*NumOfGrid - 1; i++) {
		row_ptr[i - 1] = CRS_iter;
		for (j = 1; j <= (NumOfGrid*NumOfGrid); j++) {
			if (i == j) {
				nnz[CRS_iter] = -4;
				col_ind[CRS_iter] = j - 1;
				CRS_iter += 1;
			}
			else if (((i + 1) == j) && ((i % (NumOfGrid)) != 0)) {
				if (((i - 1) % (NumOfGrid)) == 0)
					nnz[CRS_iter] = 2;
				else
					nnz[CRS_iter] = 1;
				col_ind[CRS_iter] = j - 1;
				CRS_iter += 1;
			}
			else if ((i == (j + 1)) && ((j % (NumOfGrid)) != 0)) {
				if ((i % (NumOfGrid)) == 0)
					nnz[CRS_iter] = 2;
				else
					nnz[CRS_iter] = 1;
				col_ind[CRS_iter] = j - 1;
				CRS_iter += 1;
			}
			else if ((i + NumOfGrid) == j) {
				nnz[CRS_iter] = 1;
				col_ind[CRS_iter] = j - 1;
				CRS_iter += 1;
			}
			else if (i == (j + NumOfGrid)) {
				nnz[CRS_iter] = 1;
				col_ind[CRS_iter] = j - 1;
				CRS_iter += 1;
			}
		}
	}



	row_ptr[NumOfGrid*NumOfGrid - 1] = CRS_iter;
	row_ptr[NumOfGrid*NumOfGrid] = CRS_iter + 3;
	nnz[CRS_iter] = 1;
	nnz[CRS_iter + 1] = 2;
	nnz[CRS_iter + 2] = -4;
	col_ind[CRS_iter] = NumOfGrid*NumOfGrid - NumOfGrid - 1;
	col_ind[CRS_iter + 1] = NumOfGrid*NumOfGrid - 2;
	col_ind[CRS_iter + 2] = NumOfGrid*NumOfGrid - 1;





	//CRS_pmatrix(size*size, CRS_iter, nnz, row_ptr, col_ind);
	//printf("%d\n", CRS_iter);
	
	nnz1 = (float*)malloc(sizeof(float)*(CRS_iter + 2));
	col_ind1 = (int*)malloc(sizeof(int)*(CRS_iter + 2));



	for (int i = 0; i < CRS_iter + 2; i++) {
		nnz1[i] = nnz[i];
		col_ind1[i] = col_ind[i];
	}

	int VecSizeKernel = size * sizeof(float);
	int MatSizeKernel = size * size * sizeof(float);


	cudaMalloc((void**)&res0tol, sizeof(float));
	cudaMalloc((void**)&temp1, sizeof(float));
	cudaMalloc((void**)&temp2, sizeof(float));
	cudaMalloc((void**)&alpha, sizeof(float));
	cudaMalloc((void**)&beta, sizeof(float));
	cudaMalloc((void**)&res0, VecSizeKernel);
	cudaMalloc((void**)&p0, VecSizeKernel);
	cudaMalloc((void**)&Ax, VecSizeKernel);
	cudaMalloc((void**)&Ap, VecSizeKernel);
	cudaMalloc((void**)&res1, VecSizeKernel);
	cudaMalloc((void**)&p1, VecSizeKernel);

	cudaMalloc((void**)&nnzKernel, CRS_iter + 2);
	cudaMemcpy(nnzKernel, nnz1, CRS_iter + 2, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&row_ptrKernel, size*size);
	cudaMemcpy(row_ptrKernel, row_ptr, size*size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&col_indKernel, CRS_iter + 2);
	cudaMemcpy(col_indKernel, col_ind1, CRS_iter + 2, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&rhsKernel, VecSizeKernel);
	cudaMemcpy(rhsKernel, rhs, VecSizeKernel, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&solutionKernel0, VecSizeKernel);
	cudaMemcpy(solutionKernel0, solution, VecSizeKernel, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&solutionKernel1, VecSizeKernel);
	cudaMemcpy(solutionKernel1, solution, VecSizeKernel, cudaMemcpyHostToDevice);

	LARGE_INTEGER Frequency;
	LARGE_INTEGER BeginTime;
	LARGE_INTEGER Endtime;
	cudaEvent_t start, stop;
	float time;
	CRS_iter = CRS_iter + 2;
	printf("NumOfThread\tTime for the kernel");
	int NumOfThread = 64;
	float Time;
	BOOL err;
	for (int i = 0; i < 1; i++) {
		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start, 0);

		CHECK_TIME_START;
		CRScgsolver << <(VecSizeKernel + NumOfThread - 1) / NumOfThread, NumOfThread >> >
			(size, SolutionSize, CRS_iter, nnzKernel, col_indKernel, row_ptrKernel,
			rhsKernel, solutionKernel0, solutionKernel1, res0tol, temp1, temp2, alpha, beta,
			res0, p0, Ax, Ap, res1, p1, maxiteration, tolerance);
		CHECK_TIME_END(Time, err);
		//gpuErrchk(cudaPeekAtLastError());
		//gpuErrchk(cudaDeviceSynchronize());
		//cudaEventRecord(stop, 0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&time, start, stop);

		printf("%4d\t%f\n", NumOfThread, Time);
		NumOfThread = NumOfThread + 1;
	}





	cudaMemcpy(solution, solutionKernel1, VecSizeKernel, cudaMemcpyDeviceToHost);

	cudaFree(res0);
	cudaFree(p0);
	cudaFree(res1);
	cudaFree(p1);
	cudaFree(Ax);
	cudaFree(Ap);
	cudaFree(nnzKernel);
	cudaFree(col_indKernel);
	cudaFree(row_ptrKernel);
	cudaFree(rhsKernel);
	cudaFree(solutionKernel0);
	cudaFree(solutionKernel1);
}




int main(int argc, char **argv) {
	void poissonmatrix(int, int, int, int, int, float*);
	void innerproduct(float *, float *, int, float *);
	void cgsolver(int size, float *matrix, float *rhs, float *solution, int maxiteration, float tolerance);
	void CRS_cgsolver(int size, float *rhs, float *solution, int maxiteration, float tolerance);


	int numgrid_x, numgrid_y, firstgrid_x, lastgrid_x;
	int firstrow, lastrow, matrixDOF;
	int count, ii, jj, kk, maxiteration;
	float length_x, length_y, gridsize, tolerance;
	float *rhs, *coordinate, *solution;																// CPU Variable.
	float *matrix, *Cuda_ResultVec, *CudaResultSingle;												// Cuda Variable.

	maxiteration = 20000;
	tolerance = 1.0e-10;
	gridsize = 0.08; //0.07,0.032
	length_x = 1.0;
	length_y = 1.0;
	numgrid_x = (int)round(length_x / gridsize) + 1;
	numgrid_y = (int)round(length_y / gridsize) + 1;
	matrixDOF = numgrid_x*numgrid_y;
	firstgrid_x = 1; lastgrid_x = numgrid_x;
	firstrow = 1; lastrow = matrixDOF;

	matrix = (float*)malloc(matrixDOF*matrixDOF * sizeof(float));
	coordinate = (float*)malloc(matrixDOF * 2 * sizeof(float));
	rhs = (float*)malloc(matrixDOF * sizeof(float));


	printf("================= Information ====================\n");
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	printf("CUDA : %d KB free of total %d KB\n", free/1024, total/1024);
	printf("CUDA : Maximum Grid : %d by %d \n", (int) floor(sqrt(sqrt(free/4))), (int) floor(sqrt(sqrt(free/4))));
	printf("Calculate Matrix size : %d by %d\n", numgrid_x, numgrid_y);
	printf("Calculate Vector size : %d \n", numgrid_x*numgrid_y);
	printf("Calculate CG Vec size : %d\n", numgrid_x*numgrid_y*numgrid_x*numgrid_y);
	if ((matrixDOF*matrixDOF) < (free / 4))
		printf("		   To be able to calculate matrix. \n");
	else{
		printf("error.");
		exit(-1);}
	printf("==================================================\n\n");

	printf("[Poisson] Geometry and matrix size initialized.\n");

	count = 0;
	for (ii = firstgrid_x; ii <= lastgrid_x; ii++) {
		for (jj = 1; jj <= numgrid_y; jj++) {
			coordinate[2 * count] = (ii - 1)*gridsize;
			coordinate[2 * count + 1] = (jj - 1)*gridsize;
			rhs[count] = sin(coordinate[2 * count] / length_x*PI)*sin(coordinate[2 * count + 1] / length_y*PI)*gridsize*gridsize;
			count++;}}
	printf("[Poisson] Geometry and rhs constructed.\n");
	

	for (ii = 0; ii < matrixDOF*matrixDOF; ii++)
		matrix[ii] = 0.0;
	construct_poissonmatrix(firstrow, firstgrid_x, lastgrid_x, numgrid_x, numgrid_y, matrix);
	printf("[Poisson] Poisson matrix constructed.\n");
	
	solution = (float*)malloc(matrixDOF * sizeof(float));
	for (ii = 0; ii<matrixDOF; ii++)
		solution[ii] = 1.0;

	printf("[Poisson] Start solving equations.\n");
	cgsolver(matrixDOF, matrix, rhs, solution, maxiteration, tolerance);

	
	printf("[Poisson] Poisson CRS matrix constructed.\n");
	printf("[Poisson] Start solving equations using CRS.\n");
	CRS_cgsolver(matrixDOF, rhs, solution, maxiteration, tolerance);

	


	//for (ii = 0; ii < matrixDOF; ii++)
	//	printf("%f\n", solution[ii]);
	return 0;
}