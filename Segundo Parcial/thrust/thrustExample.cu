#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <chrono>
#include <algorithm>
#include <vector>

void thrust_sequence();
void sortComparisson();
void thrustTransform();
void thrustReduce();

using sys_clock = std::chrono::system_clock;

struct functor
{
  const float a;
  functor (float _a) : a(_a) {}

  __host__ __device__ float operator()(const float &x, const float &y) const
  {
      return a * x + y;
  }
};

template <typename T>
struct square
{

  __host__ __device__ float operator()(const T & x) const
  {
      return x * x;
  }
};

struct functor_add
{
  int *data;
  functor_add(int *_data) : data(_data) {}

  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) + data[thrust::get<2>(t)];
  }
};

int main()
{

  const int size = 5;
  thrust::device_vector<int> A(size);
  thrust::device_vector<int> B(size);
  thrust::device_vector<int> res(size);

  thrust::device_vector<int> ids(size);
  thrust::device_vector<int> data(size);

  thrust::sequence(data.begin(), data.end());
  thrust::sequence(A.begin(), A.end(), 10, 10);
  thrust::sequence(B.begin(), B.end(), 5, 2);
  thrust::sequence(ids.begin(), ids.end());

  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), ids.begin(), res.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), ids.end(), res.end())),
    functor_add(thrust::raw_pointer_cast(data.data()))
  );

  thrust::host_vector<int> res_h = res;

  for(auto value : res_h)
  {
    std::cout << "result = " << value << std::endl;
  }



}

void thrustReduce()
{
  float x[4] = {1.0, 2.0 ,3.0, 4.0};

  thrust::device_vector<float> d_vec(x, x+4);

  square<float> unary_op;
  thrust::plus<float> binary_op;

  float norm = std::sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), unary_op, 0, binary_op));

  std::cout << norm << std::endl;
}

void thrustTransform()
{
  const float A = 5;
  const int size = 10;

  thrust::host_vector<float> X(size), Y(size);

  thrust::sequence(X.begin(), X.end(), 10, 10);
  thrust::sequence(Y.begin(), Y.end(), 1, 5);

  thrust::transform(X.begin(), X.end(), Y.begin(), Y.end(), functor(A));

  for(int i = 0; i < Y.size(); i++)
  {
    std::cout << "Y[" << i << "]= " << Y[i] << std::endl;
  }
}

void sortComparisson()
{
  int current_h = 0, current_d = 0, exit = 0, limit = 1 << 24;

  std::chrono::time_point<sys_clock> t1, t2;
  std::chrono::duration<double, std::milli> exec_time_ms;

  thrust::host_vector<int> h_vec(limit);

  thrust::generate(h_vec.begin(), h_vec.end(), rand);

  thrust::device_vector<int> d_vec = h_vec;

  t1 = sys_clock::now();
  thrust::sort(d_vec.begin(), d_vec.end());
  t2 = sys_clock::now();

  exec_time_ms = t2 - t1;

  printf("thrust gpu sort: %f ms/n\n", exec_time_ms.count());

  std::vector<int> stl_host_vec(h_vec.size());
  thrust::copy(h_vec.begin(), h_vec.end(), stl_host_vec.begin());

  t1 = sys_clock::now();
  std::sort(stl_host_vec.begin(), stl_host_vec.end());
  t2 = sys_clock::now();

  exec_time_ms = t2 - t1;

  printf("STL cpu sort: %f ms/n\n", exec_time_ms.count());
}

void thrust_sequence()
{
  //thrust::host_vector<int> H_vec(10, 1);
  thrust::device_vector<int> D_vec(10, 1);

  thrust::fill(D_vec.begin(), D_vec.begin() + 7, 9);

  thrust::host_vector<int> H_vec(D_vec.begin(), D_vec.begin() + 5);

  thrust::sequence(H_vec.begin(), H_vec.end(), 5, 2);

  thrust::copy(H_vec.begin(), H_vec.end(), D_vec.begin());

  int i = 0;
  for(auto value : D_vec)
  {
    printf("D[%d]= %d\n", i++, (int)value);
    //std::cout << "D[" << i++ << "]= " << value << std::endl;
  }
}
