#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "errno.h"

__global__ void multMatrix(int *a, int *b, int *c, int N)
{
	unsigned long tid = threadIdx.x;
	unsigned long bid = blockIdx.x;
	int j;
	if (tid < N)
	{
		c[N*bid+tid] = 0;
		for (j = 0; j<N; j++)
		{
			c[N*bid+tid] += (a[N*bid+j] * b[N*j+tid]);
		}
	}
}

void printMatrix(int *, int);

int main(int argc, char * argv[])
{
	int *matrizA, *matrizB, *matrizRes;
	int blocks, threads;
	int *a, *b, *c;
	int i, j, k, N;
	char *p;
	cudaEvent_t start, stop, startTotal, stopTotal;
	float tiempoTotal, tiempo;
	

	// Inicializar medidas de tiempo
	cudaEventCreate(&start);
	cudaEventCreate(&startTotal);
	cudaEventCreate(&stop);
	cudaEventCreate(&stopTotal);

	cudaEventRecord(startTotal, 0);

	if (argc >= 2)
	{
		N = (int)strtol(argv[1], &p, 10);
		if (*p != '\0' && errno != 0)
		{
			printf("Primer parametro debe ser un numero.\n");
			printf("Error: %s\n", strerror(errno));
			return 1;
		}
	}
	else
	{
		printf("No hay valor de N de entrada, ingrese valor.\n");
		return 1;
	}

	

	matrizA = (int*)malloc((N*N)*sizeof(int));
	matrizB = (int*)malloc((N*N)*sizeof(int));
	matrizRes = (int*)malloc((N*N)*sizeof(int));

	cudaMalloc((void**)&a, (N*N)*sizeof(float));
	cudaMalloc((void**)&b, (N*N)*sizeof(float));
	cudaMalloc((void**)&c, (N*N)*sizeof(float));

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			matrizA[N*i+j] = rand() % 91 + 10;
			matrizB[N*i+j] = rand() % 91 + 10;
		}
	}
	cudaMemcpy(a, matrizA, (N*N)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b, matrizB, (N*N)*sizeof(int), cudaMemcpyHostToDevice);
	
	blocks = N;
	threads = N;
	
	// Medir tiempo de calculo
	cudaEventRecord(start, 0);
	// Hacer la multiplicacion
	multMatrix<<<blocks, threads>>>(a, b, c, N);

	cudaDeviceSynchronize();
	// Finalizar tiempo de calculo
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Finalizar tiempo total
	cudaEventRecord(stopTotal, 0);
	cudaEventSynchronize(stopTotal);

	// calcular tiempos
	cudaEventElapsedTime(&tiempo, start, stop);
	cudaEventElapsedTime(&tiempoTotal, startTotal, stopTotal);

	cudaMemcpy(matrizRes, c, (N*N)*sizeof(int), cudaMemcpyDeviceToHost);

	/*printMatrix(matrizA, N);
	printMatrix(matrizB, N);
	printMatrix(matrizRes, N);*/
	
	if(argc >= 3 && !strcmp(argv[2], "2"))
	{
		printf("matriz a:\n");
		printMatrix(matrizA, N);
	

		printf("\nmatriz b:\n");
		printMatrix(matrizB, N);
	}

	// Imprimir resultado si el argumento es 1 o 2
	if(argc >= 3 && (!strcmp(argv[2], "2") || !strcmp(argv[2], "1")))
	{
	printf("\nmatriz resultado:\n");
	printMatrix(matrizRes, N);
	}



	printf("Tiempo total: %f ms.\n", tiempoTotal);
	printf("Tiempo de calculo: %f ms.\n", tiempo);

	free(matrizA);
	free(matrizB);
	free(matrizRes);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);


	return 0;
}

void printMatrix(int *matriz, int N)
{
	int i, j;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			printf("%d ", matriz[N*i+j]);
		}
		printf("\n");
	}
	printf("\n");
}