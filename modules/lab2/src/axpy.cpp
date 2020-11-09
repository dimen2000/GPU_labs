#include "../include/axpy.h"
#include <iostream>

#define MAX_SOURCE_SIZE 1024
#define GROUP_SIZE 256

void saxpy_gpu(const int vector_size, const float alpha,
  float* x, const int incx, float* y, const int incy) {
  cl_uint platformCount = 0;
  clGetPlatformIDs(0, nullptr, &platformCount);

  cl_platform_id* platforms = new cl_platform_id[platformCount];
  clGetPlatformIDs(platformCount, platforms, nullptr);

  for (cl_uint i = 0; i < platformCount; ++i) {
    char platformName[128];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
      128, platformName, nullptr);
//    std::cout << platformName << std::endl;
  }

  cl_platform_id platform = platforms[0];
  delete[] platforms;

  cl_context_properties properties[3] = {
  CL_CONTEXT_PLATFORM,(cl_context_properties)platform, 0
  };

  cl_context context = clCreateContextFromType(
  (NULL == platform) ? NULL : properties,
    CL_DEVICE_TYPE_GPU,
    NULL,
    NULL,
    NULL);

  size_t size = 0;

  clGetContextInfo(
    context,
    CL_CONTEXT_DEVICES,
    0,
    NULL,
    &size
    );

  cl_device_id device;
  if (size > 0) {
    cl_device_id* devices = (cl_device_id*)alloca(size);
    clGetContextInfo(
      context,
      CL_CONTEXT_DEVICES,
      size,
      devices,
      NULL);
    device = devices[0];
  }

  cl_command_queue queue = clCreateCommandQueue(
    context,
    device,
    0,
    NULL);

  const char fileName[] = KERNEL_PATH;
  size_t source_size;
  char* source_str;

  FILE* fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  cl_program program = clCreateProgramWithSource(
    context,
    1,
    (const char**)&source_str,
    (const size_t*)&source_size,
    NULL);

  cl_int cbp = clBuildProgram(
    program,
    1,
    &device,
    NULL,
    NULL,
    NULL);

  if (cbp < 0) {
    std::cout << "unable build program\n";
    return;
  }

  cl_kernel kernel = clCreateKernel(
    program,
    "saxpy",
    NULL);

  cl_mem input = clCreateBuffer(
    context,
    CL_MEM_READ_ONLY,
    vector_size * sizeof(float),
    NULL,
    NULL);

  cl_mem output = clCreateBuffer(
    context,
    CL_MEM_READ_WRITE,
    vector_size * sizeof(float),
    NULL,
    NULL);

  clEnqueueWriteBuffer(
    queue,
    input,
    CL_TRUE,
    0,
    vector_size * sizeof(float),
    x,
    0,
    NULL,
    NULL);

  clEnqueueWriteBuffer(
    queue,
    output,
    CL_TRUE,
    0,
    vector_size * sizeof(float),
    y,
    0,
    NULL,
    NULL);

  size_t count = vector_size;

  cl_int err = clSetKernelArg(
    kernel,
    0,
    sizeof(int),
    &count);

  err = clSetKernelArg(
    kernel,
    1,
    sizeof(float),
    &alpha);

  err = clSetKernelArg(
    kernel,
    2,
    sizeof(cl_mem),
    &input);

  err = clSetKernelArg(
    kernel,
    3,
    sizeof(int),
    &incx);

  err = clSetKernelArg(
    kernel,
    4,
    sizeof(cl_mem),
    &output);

  err = clSetKernelArg(
    kernel,
    5,
    sizeof(int),
    &incy);

  size_t group = GROUP_SIZE;

  double start_t = omp_get_wtime();

  clEnqueueNDRangeKernel(
    queue,
    kernel,
    1,
    NULL,
    &count,
    &group,
    0,
    NULL,
    NULL);

  clFinish(queue);

  double end_t = omp_get_wtime();

  std::cout << end_t - start_t << std::endl;

  clEnqueueReadBuffer(
    queue,
    output,
    CL_TRUE,
    0,
    count * sizeof(float),
    y,
    0,
    NULL,
    NULL);

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void saxpy(const int vector_size, const float alpha, float* x, const int incx, float* y, const int incy) {
  double start_t = omp_get_wtime();
  for (int i = 0; i < vector_size; ++i) {
    y[i * incy] = y[i * incy] + alpha * x[i * incx];
  }
  double end_t = omp_get_wtime();

  std::cout << end_t - start_t << std::endl;
}

void daxpy(const int vector_size, const double alpha, double* x, const int incx, double* y, const int incy) {
  double start_t = omp_get_wtime();
  for (int i = 0; i < vector_size; ++i) {
    y[i * incy] = y[i * incy] + alpha * x[i * incx];
  }
  double end_t = omp_get_wtime();

  std::cout << end_t - start_t << std::endl;
}

void saxpy_omp(const int vector_size, const float alpha, float* x, const int incx, float* y, const int incy) {
  double start_t = omp_get_wtime();
  #pragma omp parallel for
  for (int i = 0; i < vector_size; ++i) {
    y[i * incy] = y[i * incy] + alpha * x[i * incx];
  }
  double end_t = omp_get_wtime();

  std::cout << end_t - start_t << std::endl;
}

void daxpy_omp(const int vector_size, const double alpha, double* x, const int incx, double* y, const int incy) {
  double start_t = omp_get_wtime();
  #pragma omp parallel
  #pragma omp for schedule(static)
  for (int i = 0; i < vector_size; ++i) {
    y[i * incy] = y[i * incy] + alpha * x[i * incx];
  }
  double end_t = omp_get_wtime();

  std::cout << end_t - start_t << std::endl;
}

void daxpy_gpu(const int vector_size, const double alpha, double* x, const int incx, double* y, const int incy)
{
  cl_uint platformCount = 0;
  clGetPlatformIDs(0, nullptr, &platformCount);

  cl_platform_id* platforms = new cl_platform_id[platformCount];
  clGetPlatformIDs(platformCount, platforms, nullptr);

  for (cl_uint i = 0; i < platformCount; ++i) {
    char platformName[128];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
      128, platformName, nullptr);
//    std::cout << platformName << std::endl;
  }

  cl_platform_id platform = platforms[0];
  delete[] platforms;

  cl_context_properties properties[3] = {
  CL_CONTEXT_PLATFORM,(cl_context_properties)platform, 0
  };

  cl_context context = clCreateContextFromType(
  (NULL == platform) ? NULL : properties,
    CL_DEVICE_TYPE_GPU,
    NULL,
    NULL,
    NULL);

  size_t size = 0;

  clGetContextInfo(
    context,
    CL_CONTEXT_DEVICES,
    0,
    NULL,
    &size
    );

  cl_device_id device;
  if (size > 0) {
    cl_device_id* devices = (cl_device_id*)alloca(size);
    clGetContextInfo(
      context,
      CL_CONTEXT_DEVICES,
      size,
      devices,
      NULL);
    device = devices[0];
  }

  cl_command_queue queue = clCreateCommandQueue(
    context,
    device,
    0,
    NULL);

  const char fileName[] = KERNEL_PATH;
  size_t source_size;
  char* source_str;

  FILE* fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  cl_program program = clCreateProgramWithSource(
    context,
    1,
    (const char**)&source_str,
    (const size_t*)&source_size,
    NULL);

  cl_int cbp = clBuildProgram(
    program,
    1,
    &device,
    NULL,
    NULL,
    NULL);

  if (cbp < 0) {
    std::cout << "unable build program\n";
    return;
  }

  cl_kernel kernel = clCreateKernel(
    program,
    "daxpy",
    NULL);

  cl_mem input = clCreateBuffer(
    context,
    CL_MEM_READ_ONLY,
    vector_size * sizeof(double),
    NULL,
    NULL);

  cl_mem output = clCreateBuffer(
    context,
    CL_MEM_READ_WRITE,
    vector_size * sizeof(double),
    NULL,
    NULL);

  clEnqueueWriteBuffer(
    queue,
    input,
    CL_TRUE,
    0,
    vector_size * sizeof(double),
    x,
    0,
    NULL,
    NULL);

  clEnqueueWriteBuffer(
    queue,
    output,
    CL_TRUE,
    0,
    vector_size * sizeof(double),
    y,
    0,
    NULL,
    NULL);

  size_t count = vector_size;

  cl_int err = clSetKernelArg(
    kernel,
    0,
    sizeof(int),
    &count);

  err = clSetKernelArg(
    kernel,
    1,
    sizeof(double),
    &alpha);

  err = clSetKernelArg(
    kernel,
    2,
    sizeof(cl_mem),
    &input);

  err = clSetKernelArg(
    kernel,
    3,
    sizeof(int),
    &incx);

  err = clSetKernelArg(
    kernel,
    4,
    sizeof(cl_mem),
    &output);

  err = clSetKernelArg(
    kernel,
    5,
    sizeof(int),
    &incy);

  size_t group = GROUP_SIZE;

  double start_t = omp_get_wtime();

  clEnqueueNDRangeKernel(
    queue,
    kernel,
    1,
    NULL,
    &count,
    &group,
    0,
    NULL,
    NULL);

  clFinish(queue);

  double end_t = omp_get_wtime();

  std::cout << end_t - start_t << std::endl;

  clEnqueueReadBuffer(
    queue,
    output,
    CL_TRUE,
    0,
    count * sizeof(double),
    y,
    0,
    NULL,
    NULL);

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}
