/* Creado por:
** Miguel de la Colina A01021465
Practica 1 */

#include "stdlib.h"
#include "stdio.h"
#include <math.h>

#define N 100000000

__global__ void primos(char *a, int raiz)
{
    //calcular el id del thread en base a sus bloques para poder iterar por los threads
    unsigned long tid = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned long j;

    //checa que el numero de thread sea menor a la raiz de y que siga en 0 dentro del arreglo
    if(tid < raiz && a[tid] != 1)
    {
        //itera sobre todos los multiplos de tid para cambiarlos a 1 en el arreglo
        for (j=tid*tid; j<N; j+=tid)
        {
          a[j] = 1;
        }
    }
}

int main(int argc, char * argv[])
{
    // Declarar variables
    char *a;
    char *gpu_a;
    unsigned long i, c=0;
    float tiempo;
    int threads, blocks;
    cudaEvent_t start, stop;
    unsigned long raiz = sqrt(N);

    // Inicializar medidas de tiempo
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Reservar memoria en CPU
    a = (char*)malloc(N*sizeof(char));

    // Reservar memoria en GPu
    cudaMalloc((void**)&gpu_a, N*sizeof(char));

    // Llenar el arreglo en 0
    for(i = 0; i < N; ++i)
    {
        a[i] = 0;
    }
    //asignar 1 en las posiciones 1 y 0 del arreglo
    a[0] = 1;
    a[1] = 1;

    // Copiar de CPU a GPU
    cudaMemcpy(gpu_a, a, N*sizeof(char), cudaMemcpyHostToDevice);

    // Poner threads y blocks para que den 100,000,000
    threads = 1000;
    blocks = 10;

    // Hacer cálculo y medir tiempo
    cudaEventRecord(start, 0);

    //mandar a llamar la funcion con el numero de threads y de bloques
    primos<<<blocks, threads>>>(gpu_a, raiz);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiempo, start, stop);

    // Copiar de regreso a RAM de CPU
    cudaMemcpy(a, gpu_a, N*sizeof(char), cudaMemcpyDeviceToHost);


    for (i=0; i<N; i++)
    {
       if (a[i] == 0)
       {
          c++;
       }
     }

    for (i=N-1000; i<N;i++)
    {
      if(a[i] == 0)
      {
        printf("%ld ", i);
      }
    }
    printf("\n total:%ld\n", c);

    printf("El cálculo tomó %f ms\n", tiempo);

    // Liberar memoria en GPU
    cudaFree(gpu_a);

    // Liberar memoria en CPU
    free(a);

    return 0;
}
