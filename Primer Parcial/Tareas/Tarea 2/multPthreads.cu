#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "errno.h"
#include "pthread.h"
#include "math.h"

#define THREADS 4

// Recibe dos matrices y las multiplica, la matriz c debe estar inicializada en 0


typedef struct tdata
{
    unsigned long ** a;
    unsigned long ** b;
    unsigned long ** c;
    int N;
    int idx;
    int total;
}tdata;

void * matrixMult(void * data)
{
    tdata * toprocess = (tdata*)data;
    int i, j, k;
    int N = toprocess->N;
    int idx = toprocess->idx;
    int total = toprocess->total;

    for(i = 0; i < N; ++i)
    {
        if((idx*N)+i < total)
        {
            for(j = 0; j < total; ++j)
            {
                for(k = 0; k < total; ++k)
                {
                    toprocess->c[(idx*N)+i][j] += toprocess->a[(idx*N)+i][k]*toprocess->b[k][j];
                }
            }
        }
    }
    pthread_exit(NULL);

}

int main(int argc, char * argv[])
{
    // Declaracion de variables
    int i, j, cont, N, linesToDo;
    unsigned long **a, **b, **c;
    char *p;
    cudaEvent_t start, stop, startTotal, stopTotal;
    float tiempo, tiempoTotal;
    pthread_t * threads;
    tdata * data;

    // Inicializar medidas de tiempo
    cudaEventCreate(&start);
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopTotal);

    // Comenzar a medir tiempo total
    cudaEventRecord(startTotal, 0);

    if(argc >= 2)
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

    // Reservar memoria para arreglo de arreglos
    a = (unsigned long**)malloc(N*sizeof(unsigned long*));
    b = (unsigned long**)malloc(N*sizeof(unsigned long*));
    c = (unsigned long**)malloc(N*sizeof(unsigned long*));

    // Reservar memoria para cada arreglo
    for(i = 0; i < N; ++i)
    {
        *(a+i) = (unsigned long*)malloc(N*sizeof(unsigned long));
        *(b+i) = (unsigned long*)malloc(N*sizeof(unsigned long));
        *(c+i) = (unsigned long*)malloc(N*sizeof(unsigned long));
    }

    // Resrevar memoria para threads, datos de cada thread y llenarlos
    linesToDo = ceil((double)N/THREADS);

    data = (tdata*)malloc(THREADS*sizeof(tdata));
    threads = (pthread_t*)malloc(THREADS*sizeof(pthread_t));

    for(i = 0; i < THREADS; ++i)
    {
        (data+i)->a = a;
        (data+i)->b = b;
        (data+i)->c = c;
        (data+i)->N = linesToDo;
        (data+i)->idx = i;
        (data+i)->total = N;
    }


    // LLenar matrices con valores prueba
    for(i = 0; i < N; ++i)
    {
        for(j = 0; j < N; ++j)
        {
            a[i][j] = rand() % 1000 + 11;
            b[i][j] = rand() % 1000 + 11;
            c[i][j] = 0;
            ++cont;
        }
    }


    // Medir tiempo de calculo
    cudaEventRecord(start, 0);

    // Hacer la multiplicacion
    for(i = 0; i < THREADS; ++i)
    {

        if(pthread_create(threads+i, NULL, matrixMult, data+i))
        {
            printf("Error al crear thread");
            return 1;
        }
    }

    for(i = 0; i < THREADS; ++i)
    {
        pthread_join(*(threads+i), NULL);
    }

    // Finalizar tiempo de calculo
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Finalizar tiempo total
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);

    // calcular tiempos
    cudaEventElapsedTime(&tiempo, start, stop);
    cudaEventElapsedTime(&tiempoTotal, startTotal, stopTotal);

    // Imprimir que matrices se van a multiplicar
    if(argc >= 3 && !strcmp(argv[2], "2"))
    {
        printf("matriz a:\n");
        for(i = 0; i < N; ++i)
        {
            for(j = 0; j < N; ++j)
            {
                printf("%lu\t", a[i][j]);
            }
            printf("\n");
        }

        printf("\nmatriz b:\n");
        for(i = 0; i < N; ++i)
        {
            for(j = 0; j < N; ++j)
            {
                printf("%lu\t", b[i][j]);
            }
            printf("\n");
        }
    }

    // Imprimir resultado si el argumento es 1 o 2
    if(argc >= 3 && (!strcmp(argv[2], "2") || !strcmp(argv[2], "1")))
    {
        printf("\nmatriz resultado:\n");
        for(i = 0; i < N; ++i)
        {
            for(j = 0; j < N; ++j)
            {
                printf("%lu\t", c[i][j]);
            }
            printf("\n");
        }
    }


    printf("Tiempo total: %f ms.\n", tiempoTotal);
    printf("Tiempo de calculo: %f ms.\n", tiempo);

    // Liberar memoria
    for(i = 0; i < N; ++i)
    {
        free(*(a+i));
        free(*(b+i));
        free(*(c+i));
    }

    free(a);
    free(b);
    free(c);
    free(data);
    free(threads);


    return 0;
}