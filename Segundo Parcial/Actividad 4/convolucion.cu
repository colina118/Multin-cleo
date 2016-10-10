#include <stdio.h>
#include <stdlib.h>
#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>

#define Q 32
short P, M, N;

using namespace std;
using namespace cv;

void desplegar(int *m, short U, short V);

__global__ void inicializarGPU2D(int *m1, int *m2, int *m3, short p, short m, short n)
{
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   int j = blockIdx.y*blockDim.y + threadIdx.y;

   if (i < p && j < p)
      m1[i*p+j] = i+1;

   m2[i*m+j] = 255-i;
   m3[i*m+j] = 0;
}

__global__ void sumaEulerGPU(int *m1, int *m2, int *m3, short p, int m, int n)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i > 0 && j > 0 && i < n-1 && j < m-1)
  {
        m3[i*m+j] = 0;
        for (int k=0; k<p; k++)
        {
           for (int l=0; l<p; l++)
           {
              m3[i*m+j] += m2[(i-1+k)*m+j-1+l] * m1[k*p+l];
           }
        }
        m3[i*m+j] /= 9;
  }
}

void sumaEulerCPU2(int *a, int *b, int *c)
{
   for (int i=1; i<N-1; i++)
      for (int j=1; j<M-1; j++)
      {
         c[i*M+j] = 0;
         for (int k=0; k<P; k++)
            for (int l=0; l<P; l++)
               c[i*M+j] += b[(i-1+k)*M+j-1+l] * a[k*P+l];
         c[i*M+j] /= 9;
      }
}

__global__ void inicializarGPU1D(int *m1, int *m2, int *m3, short p, short m, short n)
{
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   if (i < p)
      for(int j=0; j<p; j++)
         m1[i*p+j] = i+1;

   for(int j=0; j<m; j++)
   {
      m2[i*m+j] = 255-i;
      m3[i*m+j] = 0;
   }
}

float convolucionGPU(int *a, int *b, int *c, int *a1, int d, const Mat& ima, const Mat& res)
{
   int *dev_a, *dev_b, *dev_c, *dev_a1;
   cudaEvent_t gpuI, gpuF;
   float gpuT;
   cudaEventCreate( &gpuI );
   cudaEventCreate( &gpuF );
   cudaEventRecord( gpuI, 0 );

   cudaMalloc( (void**)&dev_a, 3*3*sizeof(int) );
   cudaMalloc( (void**)&dev_a1, 3*3*sizeof(int) );
   cudaMalloc( (void**)&dev_b, ima.step*ima.rows*sizeof(int) );
   cudaMalloc( (void**)&dev_c, res.step*res.rows*sizeof(int) );

   cudaMemcpy( dev_a, a, 3*3*sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpy( dev_a1, a1, 3*3*sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpy( dev_b, b, ima.step*ima.rows*sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpy( dev_c, c, res.step*res.rows*sizeof(int), cudaMemcpyHostToDevice );

   if (d == 1)

   else if (d == 2)
   {
      dim3 bloques( ima.rows/Q, ima.step/Q);
      dim3 threads( Q, Q);
      sumaEulerGPU<<<bloques, threads>>>(dev_a, dev_b, dev_c, 3, ima.step, ima.rows);
   }
   cudaDeviceSynchronize();
   //cudaMemcpy( a, dev_a, P*P*sizeof(int), cudaMemcpyDeviceToHost );
   cudaMemcpy( b, dev_b, ima.step*ima.rows*sizeof(int), cudaMemcpyDeviceToHost );
   cudaMemcpy( c, dev_c, res.step*res.rows*sizeof(int), cudaMemcpyDeviceToHost );

   cudaEventRecord( gpuF, 0 );
   cudaEventSynchronize( gpuF );
   cudaEventElapsedTime( &gpuT, gpuI, gpuF );
   cudaFree( dev_a );
   cudaFree( dev_b );
   cudaFree( dev_c );
   return gpuT;
}

void sumaEulerCPU1(int *a, int *b, int *c)
{
   // Se calculan valores sin considerar la orilla para 3x3.
   for (int i=1; i<N-1; i++)
      for (int j=1; j<M-1; j++)
         c[i*M+j] = ( b[i*M+j] * a[4] + b[(i-1)*M+j-1] * a[0] + b[(i-1)*M+j] * a[1] + b[(i-1)*M+j+1] * a[2] + b[i*M+j-1] * a[3] + b[i*M+j+1] * a[5] + b[(i+1)*M+j-1] * a[6] + b[(i+1)*M+j] * a[7] + b[(i+1)*M+j+1] * a[8] ) / 9;
         // En medio, esq sup izq, arriba, esq sup der, izq, der, esq inf izq, abajo, esq inf der
}



float convolucionCPU1(int *a, int *b, int *c)
{
   cudaEvent_t cpuI, cpuF;
   float cpuT;
   cudaEventCreate( &cpuI );
   cudaEventCreate( &cpuF );
   cudaEventRecord( cpuI, 0 );

   sumaEulerCPU1(a, b, c);

   cudaEventRecord( cpuF, 0 );
   cudaEventSynchronize( cpuF );
   cudaEventElapsedTime( &cpuT, cpuI, cpuF );
   return cpuT;
}

float convolucionCPU2(int *a, int *b, int *c)
{
   cudaEvent_t cpuI, cpuF;
   float cpuT;
   cudaEventCreate( &cpuI );
   cudaEventCreate( &cpuF );
   cudaEventRecord( cpuI, 0 );

   sumaEulerCPU2(a, b, c);

   cudaEventRecord( cpuF, 0 );
   cudaEventSynchronize( cpuF );
   cudaEventElapsedTime( &cpuT, cpuI, cpuF );
   return cpuT;
}

void desplegar(int *m, short U, short V)
{
   for (int i=0; i<V; i++)
   {
      for (int j=0; j<U; j++)
         printf("%d ", m[i*U+j]);
      printf("\n");
   }
   printf("\n");
}

void inicializar(int *a, int *b, int *c)
{
   short valor = 255;

   for(int i=0; i<P; i++)
      for(int j=0; j<P; j++)
         a[i*P+j] = i+1;

   for(int i=0; i<N; i++)
   {
      for(int j=0; j<M; j++)
      {
         b[i*M+j] = valor;
         c[i*M+j] = 0;
      }
      valor--;
      if (valor < 0)
         valor = 255;
   }
}

void calcular( char o, char p, char d, const Mat& ima, const Mat& res)
{
   int *masPxP, *resMxN, *imaMxN, *masPyP;
   masPxP = (int*) malloc(P*P*sizeof(int));
   masPxP = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
   masPyP = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
   imaMxN = (int*) malloc(ima.step*ima.rows*sizeof(int));
   resMxN = (int*) malloc(res.step*res.rows*sizeof(int));
   memcpy(imaMxN, ima.ptr(), ima.step*ima.rows);
   memcpy(resMxN, res.ptr(), res.step*res.rows);


   inicializar( masPxP, imaMxN, resMxN );
   if (p)
   {
      printf("Matrix A (máscara) de %dx%d\n", P, P);
      desplegar(masPxP, P, P);
      printf("Matrix B de %dx%d\n", M, N);
      desplegar(imaMxN, M, N);
      printf("Matrix C (resultado inicializado) de %dx%d\n", M, N);
      desplegar(resMxN, M, N);
   }

   if (o == 0)
   {
      printf("Tiempo (CPU) forma1: %f ms\n", convolucionCPU1( masPxP, imaMxN, resMxN ) );
      if (p)
      {
         printf("Matrix C (resultado) de %dx%d\n", M, N);
         desplegar(resMxN, M, N);
      }
      printf("Tiempo (CPU) forma2: %f ms\n", convolucionCPU2( masPxP, imaMxN, resMxN ) );
      if (p)
      {
         printf("Matrix C (resultado) de %dx%d\n", M, N);
         desplegar(resMxN, M, N);
      }
   }
   else if (o == 1)
   {
      printf("Tiempo (GPU): %f ms\n", convolucionGPU( masPxP, imaMxN, resMxN, d, ima, res) );
      if (p)
      {
         printf("Matrix A (máscara) de %dx%d\n", M, N);
         desplegar(masPxP, P, P);
         printf("Matrix B de %dx%d\n", M, N);
         desplegar(imaMxN, M, N);
         printf("Matrix C (resultado inicializado) de %dx%d\n", M, N);
         desplegar(resMxN, M, N);
      }
   }
   free( masPxP );
   free( imaMxN );
   free( resMxN );
}

int main (int argc, char *argv[] )
{
   if ( argc != 7 )
   {
      printf("%s P M N 0|1(CPU|GPU) 0|1(no|despliega) 1|2|3(D)\n", argv[0]);
      exit(0);
   }
   P = atoi(argv[1]);
   M = atoi(argv[2]);
   N = atoi(argv[3]);
   if ( P != 3 && P != 5 && P != 7 )
   {
      printf("P debe ser valor impar entre 3 y 7\n");
      exit(0);
   }
   if ( M%512 != 0 || M < 512 || M > 2048 )
   {
      printf("M debe ser valor múltiplo de 512 entre 512 y 2048\n");
      exit(0);
   }
   if ( N%256 != 0 || M < 256 || M > 1024 )
   {
      printf("N debe ser valor múltiplo de 256 entre 256 y 1024\n");
      exit(0);
   }
  string imagePath;

   if(argc < 2)
  	imagePath = "space-wallpaper_2880x1800.jpg";
  else
  	imagePath = argv[1];

  //string imagePath = "samurai-girl-pictures_2560x1600.jpg";
  //string imagePath = "women_samurai_2560x1600.jpg";
  //string imagePath = "space-wallpaper_2880x1800.jpg";

  //Read input image from the disk
  Mat input = imread(imagePath, CV_LOAD_IMAGE_GRAYSCASLE);
  Mat output = imread (imagePath, CV_LOAD_IMAGE_GRAYSCASLE);
  //Mat output(input.rows, input.cols, CV_8UC1);

  if (input.empty())
  {
  	cout << "Image Not Found!" << std::endl;
  	cin.get();
  	return -1;
  }


   calcular( atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), input, output;
   return 1;
}
