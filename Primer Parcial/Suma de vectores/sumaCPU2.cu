/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
#include <stdio.h>

#define N   1000000

void add( int *a, int *b, int *c ) {
   int tid = 0;    // this is CPU zero, so we start at zero
   while (tid < N) {
      c[tid] = a[tid] + b[tid];
      tid ++;   // we have one CPU, so we increment by one
   }
}

int main( void ) {
   int *a=new int[N], *b = new int[N], *c = new int[N];
   cudaEvent_t inicio, fin;
   float tiempo;

   // fill the arrays 'a' and 'b' on the CPU
   for (int i=0; i<N; i++)
      a[i] = b[i] = i+1;

   cudaEventCreate( &inicio );
   cudaEventCreate( &fin );
   cudaEventRecord( inicio, 0 );
   add( a, b, c );
   cudaEventRecord( fin, 0 );
   cudaEventSynchronize( fin );
   cudaEventElapsedTime( &tiempo, inicio, fin );

   // display the results
  // for (int i=0; i<N; i++)
   //   printf( "%d + %d = %d\n", a[i], b[i], c[i] );

   free(a);
   free(b);
   free(c);

   printf("tiempo total en ms: %f\n", tiempo);

   return 0;
}
