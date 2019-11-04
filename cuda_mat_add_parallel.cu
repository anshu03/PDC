#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void gpu_matrixadd(int *a,int *b, int *c, int N) {

	int col = threadIdx.x + blockDim.x * blockIdx.x; 
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int index = row * N + col;

      	if(col < N && row < N)
          c[index] = a[index]+b[index];

}

void cpu_matrixadd(int *a,int *b, int *c, int N) {

	int index;
	for(int col=0;col < N; col++) 
		for(int row=0;row < N; row++) {
			index = row * N + col;
           		c[index] = a[index]+b[index];
		}
}

int main(int argc, char *argv[])  {

	char key;

	int i, j;

	int Grid_Dim_x=1, Grid_Dim_y=1;
	int Block_Dim_x=1, Block_Dim_y=1;

	int noThreads_x, noThreads_y;
	int noThreads_block;

	int N = 10;
	int *a,*b,*c,*d;
	int *dev_a, *dev_b, *dev_c;
	int size;

	cudaEvent_t start, stop; 
	float elapsed_time_ms;


do {

	__global__ void input_parameter(sizeof(x), sizeof(y), nub_block))

		x = (int*) malloc(size);
		y = (int*) malloc(size);
		nub_block = (int*) malloc(size);

	for(i=0;i < N;i++)
	for(j=0;j < N;j++) {
		a[i * N + j] = i;
		b[i * N + j] = i;
	}

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	cudaMemcpy(dev_a, a , size ,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b , size ,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c , size ,cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
//	cudaEventSynchronize(start);  	// Needed?

	gpu_matrixadd<<<Grid,Block>>>(dev_a,dev_b,dev_c,N);

	cudaMemcpy(c,dev_c, size ,cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );

//	for(i=0;i < N;i++) 
//	for(j=0;j < N;j++)
//	   printf("%d+%d=%d\n",a[i * N + j],b[i * N + j],c[i * N + j]);
	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);  // print out execution time


	cudaEventRecord(start, 0);
//	cudaEventSynchronize(start);  	// Needed?

	cpu_matrixadd(a,b,d,N);		// do calculation on host

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );

	printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms);  // print out execution time
}

