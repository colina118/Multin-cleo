#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "errno.h"

__global__ void convolucion(int *a, int *b, int *c, int N, int M, int P)
{
	unsigned long tid = threadIdx.x;
  unsigned long bid = blockIdx.x;
	int j, k;
  int res =0;
	if (tid+P < N && bid+P < M)
	{
    for(j = 0; j < P; j++)
    {
      for(k = 0; k < P; k++)
      {
          res += (a[P*j+k] * b[N*(tid+j)+(bid+k)]);
      }
    }
	}

  res = res/9;
  if(P == 7)
  {
      c[N*(tid+3)+(bid+3)] = res;
  }
  else if(P == 5)
  {
    c[N*(tid+2)+(bid+2)] = res;
  }
  else if(P == 3)
  {
    c[N*(tid+1)+(bid+1)] = res;
  }

}

void printMatrix(int *, int, int);

int main(int argc, char * argv[])
{
	int *matrizA, *matrizB, *matrizRes;
	int blocks, threads;
	int *a, *b, *c;
	int i, j, N, M, P, cont;
	char *p;
	cudaEvent_t start, stop, startTotal, stopTotal;
	float tiempoTotal, tiempo;
  cont = 255;

	// Inicializar medidas de tiempo
	cudaEventCreate(&start);
	cudaEventCreate(&startTotal);
	cudaEventCreate(&stop);
	cudaEventCreate(&stopTotal);

	cudaEventRecord(startTotal, 0);

	if (argc >= 3)
	{
		N = (int)strtol(argv[1], &p, 10);
    M = (int)strtol(argv[2], &p, 10);
    P = (int)strtol(argv[3], &p, 10);
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

  printf("M %d, N %d, P %d\n", M, N, P);

  if(P%2 == 0)
  {
    printf("La matriz P no es multiplo de 3\n");
    return 0;
  }
  else if(P > 7)
  {
    printf("La matriz P es mayoer que 7x7\n");
    return 0;
  }
  else if(P < 3)
  {
    printf("La matriz P es menor que 3x3\n");
    return 0;
  }

  if(M % 512 != 0)
  {
    printf("M no es multiplo de 512\n");
    return 0;
  }
  else if(M < 512)
  {
    printf("M es menor que 512\n");
    return 0;
  }
  else if(M > 2048)
  {
    printf("M es mayor que 2048");
    return 0;
  }

  if(N % 256 != 0)
  {
    printf("N no es multiplo de 512\n");
    return 0;
  }
  else if(N < 256)
  {
    printf("N es menor que 512\n");
    return 0;
  }
  else if(N > 1024)
  {
    printf("N es mayor que 2048");
    return 0;
  }



	matrizA = (int*)malloc((P*P)*sizeof(int));
	matrizB = (int*)malloc((M*N)*sizeof(int));
	matrizRes = (int*)malloc((M*N)*sizeof(int));

	cudaMalloc((void**)&a, (P*P)*sizeof(float));
	cudaMalloc((void**)&b, (M*N)*sizeof(float));
	cudaMalloc((void**)&c, (M*N)*sizeof(float));

	for (i = 0; i < P; i++)
	{
		for (j = 0; j < P; j++)
		{
			matrizA[P*i+j] = i+1;
		}
	}

  for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			matrizB[N*i+j] = cont;
      matrizRes[N*i+j] = cont;
		}
    cont --;
    if(cont == 0)
    {
      cont = 255;
    }
	}

	cudaMemcpy(a, matrizA, (P*P)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b, matrizB, (M*N)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c, matrizRes, (M*N)*sizeof(int), cudaMemcpyHostToDevice);

	blocks = M;
	threads = N;

	// Medir tiempo de calculo
	cudaEventRecord(start, 0);
	// Hacer la multiplicacion
	convolucion<<<blocks, threads>>>(a, b, c, N, M, P);

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

	cudaMemcpy(matrizRes, c, (M*N)*sizeof(int), cudaMemcpyDeviceToHost);

	/*printMatrix(matrizA, N);
	printMatrix(matrizB, N);
	printMatrix(matrizRes, N);*/
  printMatrix(matrizA, P, P);
  printMatrix(matrizB, M, N);
  printMatrix(matrizRes, M, N);

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

void printMatrix(int *matriz, int N, int M)
{
	int i, j;
	for (i = N-4; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			printf("%d ", matriz[M*i+j]);
		}
		printf("\n");
	}
	printf("\n");
}
