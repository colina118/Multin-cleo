#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>

using namespace std;
using namespace cv;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


__global__ void bgr_to_gray_kernel(unsigned char* input,
	unsigned char* output,
	int width,
	int height,
	int colorWidthStep,
	int grayWidthStep)
{
	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		//Location of gray pixel in output
		const int gray_tid = yIndex * grayWidthStep + xIndex;

		const unsigned char blue = input[color_tid];
		const unsigned char green = input[color_tid + 1];
		const unsigned char red = input[color_tid + 2];

		//then you can use the standard NTSC conversion formula that is used for calculating the effective luminance of a pixel :
		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;
		//const float gray = (red + green + blue) / 3.f;

		output[gray_tid] = static_cast<unsigned char>(gray);
	}
}

void convert_to_gray(const Mat& input, Mat& output)
{
	//Calculate total number of bytes of input and output image
	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	//Launch the color conversion kernel
	bgr_to_gray_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), static_cast<int>(output.step));

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[])
{
	string imagePath;

	if(argc < 2)
		imagePath = "space-wallpaper_2880x1800.jpg";
  else
  	imagePath = argv[1];

	//string imagePath = "samurai-girl-pictures_2560x1600.jpg";
	//string imagePath = "women_samurai_2560x1600.jpg";
  //string imagePath = "space-wallpaper_2880x1800.jpg";

	//Read input image from the disk
	Mat input = imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
	Mat output(input.rows, input.cols, CV_8UC1);

	//Call the wrapper function
	convert_to_gray(input, output);

	//Allow the windows to resize
	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("Output", WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);

	//Wait for key press
	waitKey();

	return 0;
}
