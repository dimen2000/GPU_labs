#include <CL/cl.h>
#include <iostream>
#include <random>
#include <ctime>
#include "../include/axpy.h"

#define MAX_SOURCE_SIZE 1024
#define SIZE (1 << 28)
//#define SAXPY

int main() {

  int a = 1;
  int ix = 1;
  int iy = 1;


#ifdef SAXPY
  std::cout << "saxpy\n\n";

  float* x = new float[SIZE];
  float* y = new float[SIZE];
  float* y1 = new float[SIZE];
  float* y2 = new float[SIZE];

  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  for (int i = 0; i < SIZE; i++) {
    x[i] = gen();
    y[i] = gen();
    y1[i] = y[i];
    y2[i] = y[i];
  }

  std::cout << "sequental\n\n";

  saxpy(SIZE, a, x, ix, y, iy);

  std::cout << "\n\ngpu\n\n";

  saxpy_gpu(SIZE, a, x, ix, y1, iy);

  for (int i = 0; i < SIZE; i++) {
    if (y[i] != y1[i])
      std::cout << "unequality!\n";
  }

  std::cout << "\n\nomp\n\n";

  saxpy_omp(SIZE, a, x, ix, y2, iy);

  for (int i = 0; i < SIZE; i++) {
    if (y[i] != y2[i])
      std::cout << "unequality!\n";
  }

#endif
#ifndef SAXPY

  double* x = new double[SIZE];
  double* y = new double[SIZE];
  double* y1 = new double[SIZE];
  double* y2 = new double[SIZE];

  std::cout << "daxpy\n\n";

  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  for (int i = 0; i < SIZE; i++) {
    x[i] = gen();
    y[i] = gen();
    y1[i] = y[i];
    y2[i] = y[i];
  }

  std::cout << "sequental\n\n";

  daxpy(SIZE, a, x, ix, y, iy);

  std::cout << "\n\ngpu\n\n";

  daxpy_gpu(SIZE, a, x, ix, y1, iy);

  for (int i = 0; i < SIZE; i++) {
    if (y[i] != y1[i])
      std::cout << "unequality!\n";
  }

  std::cout << "\n\nomp\n\n";

  daxpy_omp(SIZE, a, x, ix, y2, iy);

  for (int i = 0; i < SIZE; i++) {
    if (y[i] != y2[i])
      std::cout << "unequality!\n";
  }
#endif
  delete[]x;
  delete[]y;
  delete[]y1;
  delete[]y2;
  return 0;
}