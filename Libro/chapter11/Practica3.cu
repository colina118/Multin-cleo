#include <stdio.h>
#include "../common/book.h"

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

struct DataStruct {
    int     deviceID;
    int     size;
    int   *a;
    int   *b;
    int *c;
    int   returnC;
};

void* sumaGPUs(void *voiddata)
{
   DataStruct  *data = (DataStruct*)voiddata;
   HANDLE_ERROR( cudaSetDevice( data->deviceID ) );

   int   size = data->size;
   int   *a, *b, *c;
   int   *d_a, *d_b, *d_c;
   float tiempo1, tiempo2;
   cudaEvent_t inicio1, fin1, inicio2, fin2;

   /* allocate space for host copies of a, b, c and setup input alues */

   a = data->a;
   b = data->b;
   c = data->c;

   cudaEventCreate(&inicio1); // Se inicializan
   cudaEventCreate(&fin1);
   cudaEventRecord( inicio1, 0 ); // Se toma el tiempo de inicio

   /* allocate space for device copies of a, b, c */

   cudaMalloc( (void **) &d_a, size * sizeof(int));
   cudaMalloc( (void **) &d_b, size * sizeof(int));
   cudaMalloc( (void **) &d_c, size * sizeof(int));

   /* copy inputs to deice */
   /* fix the parameters needed to copy data to the device */
   cudaMemcpy( d_a, a, size* sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpy( d_b, b, size* sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpy( d_c, c, size* sizeof(int), cudaMemcpyHostToDevice );

   cudaEventCreate(&inicio2); // Se inicializan
   cudaEventCreate(&fin2);
   cudaEventRecord( inicio2, 0 ); // Se toma el tiempo de inicio

   /* launch the kernel on the GPU */
   add<<< size/ THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );

   cudaEventRecord( fin2, 0); // Se toma el tiempo final.
   cudaEventSynchronize( fin2 ); // Se sincroniza
   cudaEventElapsedTime( &tiempo2, inicio2, fin2 );

   /* copy result back to host */
   /* fix the parameters needed to copy data back to the host */
   cudaMemcpy( c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost );

   data->c = c;
   for(int i = 0; i < 10; i++)
   {
     printf("%d", data->c[i]);
   }

   cudaFree( d_a );
   cudaFree( d_b );
   cudaFree( d_c );

   cudaEventRecord( fin1, 0); // Se toma el tiempo final.
   cudaEventSynchronize( fin1 ); // Se sincroniza
   cudaEventElapsedTime( &tiempo1, inicio1, fin1 );

   printf("Tiempo cálculo %f ms\n", tiempo2);
   printf("Tiempo total %f ms\n", tiempo1);

   return 0;
} /* end main */

int main()
{
  cudaDeviceProp prop;
  int deviceCount=0;
  cudaGetDeviceCount( &deviceCount );
  for(int i =0; i<deviceCount; i++)
  {
    cudaGetDeviceProperties(&prop, i);
    printf("num:%d, nombre:%s\n", i, prop.name);
  }
  if (deviceCount < 2) {
      printf( "We need at least two compute 1.0 or greater "
              "devices, but only found %d\n", deviceCount );
      return 0;
  }

  int *a, *b, *c;

  a=(int*)malloc(sizeof(int) * N );
  b=(int*)malloc(sizeof(int) * N );
  c=(int*)malloc(sizeof(int) * N );

  for( int i = 0; i < N; i++ )
  {
    a[i] = b[i] = i+1;
    c[i] = 0;
  }


  // prepare for multithread
  DataStruct  data[2];
  data[0].deviceID = 0;
  data[0].size = (N*3)/4;
  data[0].a = a;
  data[0].b = b;
  data[0].c = c;

  data[1].deviceID = 1;
  data[1].size = N/4;
  data[1].a = a + ((N*3)/4);
  data[1].b = b + ((N*3)/4);
  data[1].c = c + ((N*3)/4);

  CUTThread   thread = start_thread( sumaGPUs, &(data[0]) );
  sumaGPUs( &(data[1]) );
  end_thread( thread );
  int j = 0;
  for(int i = 0; i <(N*3)/4; i++)
  {
      c[i] = data[0].c[i];
  }
  for(int i = (N*3)/4; i < N; i++)
  {
    c[i] = data[1].c[j];
    j++;
  }

  for(int i = 0; i<N; i++)
  {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  free( a );
  free( b );
  free(c);

  return 0;
}
