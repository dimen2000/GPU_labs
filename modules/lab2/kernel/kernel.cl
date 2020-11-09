__kernel void saxpy(const int n, const float a, global float* x, const int incx, global float* y, const int incy) {
    int i = get_global_id(0);
	
	if(i < n)
		y[i * incy] = y[i * incy] + a * x[i * incx];
}

__kernel void daxpy(const int n, const double a, global double* x, const int incx, global double* y, const int incy) {
    int i = get_global_id(0);
	
	if(i < n)
		y[i * incy] = y[i * incy] + a * x[i * incx];
}