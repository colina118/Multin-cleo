#include <stdio.h>

/* experiment with N */
/* how large can it be? */
#define N 1000000
#define THREADS_PER_BLOCK 1000

__global__ void add(int *a, int *b, int *c)
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < N)
      c[index] = a[index] + b[index];
}

int main()
{
   int *a, *b, *c;
   int *d_a, *d_b, *d_c;
   int size = N * sizeof( int );
   float tiempo1, tiempo2;
   cudaEvent_t inicio1, fin1, inicio2, fin2; // para medir tiempos como con timestamp

   /* allocate space for host copies of a, b, c and setup input alues */

   a = (int *)malloc( size );
   b = (int *)malloc( size );
   c = (int *)malloc( size );

   for( int i = 0; i < N; i++ )
      a[i] = b[i] = i+1;

   cudaEventCreate(&inicio1); // Se inicializan
   cudaEventCreate(&fin1);
   cudaEventRecord( inicio1, 0 ); // Se toma el tiempo de inicio

   /* allocate space for device copies of a, b, c */

   cudaMalloc( (void **) &d_a, size );
   cudaMalloc( (void **) &d_b, size );
   cudaMalloc( (void **) &d_c, size );

   /* copy inputs to deice */
   /* fix the parameters needed to copy data to the device */
   cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
   cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

   cudaEventCreate(&inicio2); // Se inicializan
   cudaEventCreate(&fin2);
   cudaEventRecord( inicio2, 0 ); // Se toma el tiempo de inicio

   /* launch the kernel on the GPU */
   add<<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );

   cudaEventRecord( fin2, 0); // Se toma el tiempo final.
   cudaEventSynchronize( fin2 ); // Se sincroniza
   cudaEventElapsedTime( &tiempo2, inicio2, fin2 );

   /* copy result back to host */
   /* fix the parameters needed to copy data back to the host */
   cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );

   cudaFree( d_a );
   cudaFree( d_b );
   cudaFree( d_c );

   cudaEventRecord( fin1, 0); // Se toma el tiempo final.
   cudaEventSynchronize( fin1 ); // Se sincroniza
   cudaEventElapsedTime( &tiempo1, inicio1, fin1 );

   for (int i=0; i<N; i++)
      printf( "%d + %d = %d\n", a[i], b[i], c[i] );

   /* clean up */

   free(a);
   free(b);
   free(c);

   printf("Tiempo cálculo %f ms\n", tiempo2);
   printf("Tiempo total %f ms\n", tiempo1);
	
   return 0;
} /* end main */
