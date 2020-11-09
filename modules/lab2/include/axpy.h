#include <CL/cl.h>
#include <omp.h>


void saxpy_gpu(const int vector_size, const float alpha,
  float* x, const int incx, float* y, const int incy);
void saxpy(const int vector_size, const float alpha,
  float* x, const int incx, float* y, const int incy);
void daxpy(const int vector_size, const double alpha,
  double* x, const int incx, double* y, const int incy);
void saxpy_omp(const int vector_size, const float alpha,
  float* x, const int incx, float* y, const int incy);
void daxpy_omp(const int vector_size, const double alpha,
  double* x, const int incx, double* y, const int incy);
void daxpy_gpu(const int vector_size, const double alpha,
  double* x, const int incx, double* y, const int incy);