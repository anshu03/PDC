#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void gpu_ MatrixMul(int *a,int *b, int *c, int N) {

	int col = threadIdx.x + blockDim.x * blockIdx.x; 
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int index = row * N + col;

      	if(col < N && row < N)
          c[index] = a[index]+b[index];

}

void cpu_ MatrixMul (int *a,int *b, int *c, int N) {

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

	printf ("Device characteristics -- some limitations (compute capability 1.0)\n");
	printf ("		Maximum number of threads per block = 512\n");
	printf ("		Maximum sizes of x- and y- dimension of thread block = 512\n");
	printf ("		Maximum size of each dimension of grid of thread blocks = 65535\n");
	printf("Enter size of array in one dimension (square array), currently %d\n",N);
	scanf("%d",&N);
	do {
		printf("\nEnter nuumber of blocks per grid in x dimension), currently %d  : ",Grid_Dim_x);
		scanf("%d",&Grid_Dim_x);

		printf("\nEnter nuumber of blocks per grid in y dimension), currently %d  : ",Grid_Dim_y);
		scanf("%d",&Grid_Dim_y);

		printf("\nEnter nuumber of threads per block in x dimension), currently %d  : ",Block_Dim_x);
		scanf("%d",&Block_Dim_x);

		printf("\nEnter nuumber of threads per block in y dimension), currently %d  : ",Block_Dim_y);
		scanf("%d",&Block_Dim_y);

		noThreads_x = Grid_Dim_x * Block_Dim_x;
		noThreads_y = Grid_Dim_y * Block_Dim_y;

		noThreads_block = Block_Dim_x * Block_Dim_y;

		if (noThreads_x < N) printf("Error -- number of threads in x dimension less than number of elements in arrays, try again\n");
		else if (noThreads_y < N) printf("Error -- number of threads in y dimension less than number of elements in arrays, try again\n");
		else if (noThreads_block > 512) printf("Error -- too many threads in block, try again\n");
		else printf("Number of threads not used = %d\n", noThreads_x * noThreads_y - N * N);

	} while (noThreads_x < N || noThreads_y < N || noThreads_block > 512);

	dim3 Grid(Grid_Dim_x, Grid_Dim_x);
	dim3 Block(Block_Dim_x,Block_Dim_y);

	size = N * N * sizeof(int);

	a = (int*) malloc(size);
	b = (int*) malloc(size);
	c = (int*) malloc(size);
	d = (int*) malloc(size);

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

	cpu_matrixadd(a,b,d,N);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );

	printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms);


	for(i=0;i < N*N;i++) {
		if (c[i] != d[i]) printf("*********** ERROR in results, CPU and GPU create different answers ********\n");
		break;
	}

	printf("\nEnter c to repeat, return to terminate\n");
	scanf("%c",&key);
	scanf("%c",&key);

} while (key == 'c');

	free(a);
	free(b);
	free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
