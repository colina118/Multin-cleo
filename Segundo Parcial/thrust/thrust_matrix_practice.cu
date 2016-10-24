#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using sys_clock = std::chrono::system_clock;

/// used to fill a host vector
struct rand_functor
{
	int mod = 0;
	rand_functor(int _mod = 0) : mod(_mod) { std::srand(std::time(0)); }

	template<typename T>
	void operator()(T &var)
	{
		if(mod > 0)
			var = std::rand() % mod;
		else
			var = std::rand();
	}
};

struct matrix_mult
{
	float *data;
	float *datab;
	int N;
  matrix_mult(float *_data, float *_datab, int _N) : data(_data), datab(_datab), N(_N) {}

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
		for (int j = 0; j<N; j++)
		{
			thrust::get<0>(t) += (data[N*thrust::get<1>(t)+j] * datab[N*j+thrust::get<2>(t)]);
		}
  }
};

void cpu_matrix_mult(float *A, float *B, float *C, int row_size, int col_size)
{
	int i, j, k;

	for(i = 0; i < row_size; ++i)
	{
			for(j = 0; j < row_size; ++j)
			{
					for(k = 0; k < col_size; ++k)
					{
							C[row_size * i +j] += A[row_size * i + k] * B[row_size * k + j];
					}
			}
	}
}

void print_matrix(float *A, int row_size, int col_size)
{
	std::cout << "\n";
	for(int i = 0; i < row_size; i++)
	{
		for(int j = 0; j <col_size; j++)
		{
			std::cout << A[i * col_size + j] << " ";
		}
		std::cout << "\n";
	}
}

void thrust_matrix_mult(const int row_size, const int col_size)
{
	const int matrix_size = col_size * row_size;

	std::chrono::time_point<sys_clock> t1, t2;
	std::chrono::duration<double, std::milli> exec_time_ms;

	/// These are for the CPU matrix mult
	float *A = (float*)malloc(sizeof(float) * matrix_size);
	float *B = (float*)malloc(sizeof(float) * matrix_size);
	float *C = (float*)malloc(sizeof(float) * matrix_size);

	/// Vectors for the thrust matrix mult
	thrust::host_vector<float> result(matrix_size);
	thrust::host_vector<float> matrix_hA(matrix_size), matrix_hB(matrix_size);
	thrust::device_vector<float> matrix_A(matrix_size), matrix_B(matrix_size), matrix_C(matrix_size, 0.0f);

	thrust::device_vector<int> ids;
	thrust::device_vector<int> ids2;

	/// Additional variables you may need

	thrust::for_each(matrix_hA.begin(), matrix_hA.end(), rand_functor(10));
	thrust::for_each(matrix_hB.begin(), matrix_hB.end(), rand_functor(10));
	//thrust::sequence(ids.begin(), ids.end());
	for(int i = 0; i< row_size; i++)
	{
		for(int j = 0; j < col_size; j++)
		{
			ids2.push_back(j);
			ids.push_back(i);
		}
	}



	matrix_A = matrix_hA;
	matrix_B = matrix_hB;

	thrust::copy(matrix_A.begin(), matrix_A.end(), A);
	thrust::copy(matrix_B.begin(), matrix_B.end(), B);

	t1 = sys_clock::now();
	cpu_matrix_mult(A, B, C, row_size, col_size);
	t2 = sys_clock::now();

	exec_time_ms = t2 - t1;

	std::cout << "CPU mm time: " << exec_time_ms.count() << "ms\n";

	t1 = sys_clock::now();

	thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(matrix_C.begin(), ids.begin(), ids2.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(matrix_C.end(), ids.end(), ids2.end())),
    matrix_mult(thrust::raw_pointer_cast(matrix_A.data()), thrust::raw_pointer_cast(matrix_B.data()), row_size)
  );

	result = matrix_C;
	t2 = sys_clock::now();

	exec_time_ms = t2 - t1;
	std::cout << "Thrust GPU mm time: " << exec_time_ms.count() << "ms\n";

	std::cout << "\nChecking Matrices" << std::endl;
	bool funciona = true;
  for(int i = 0; i < matrix_size; i++)
	{
		if(result[i] != C[i])
		{
			funciona = false;
			break;
		}

	}

	if(funciona)
	{
		printf("Si te funciono papawh\n");
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2)
		thrust_matrix_mult(50, 50);
	else
		thrust_matrix_mult(atoi(argv[1]), atoi(argv[1]));
	return 0;
}
