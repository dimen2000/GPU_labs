#include <CL/cl.h>
#include <iostream>

#define MAX_SOURCE_SIZE 1024
#define SIZE 1024

int main() {
  cl_uint platformCount = 0;
  clGetPlatformIDs(0, nullptr, &platformCount);

  cl_platform_id* platforms = new cl_platform_id[platformCount];
  clGetPlatformIDs(platformCount, platforms, nullptr);

  for (cl_uint i = 0; i < platformCount; ++i) {
    char platformName[128];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
      128, platformName, nullptr);
    std::cout << platformName << std::endl;
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

  char fileName[] = KERNEL_PATH;
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

  cl_kernel kernel = clCreateKernel(
    program,
    "test",
    NULL);


  int data[SIZE];
  int result[SIZE];

  for (int i = 0; i < SIZE; ++i) {
    data[i] = rand();
  }

  cl_mem input = clCreateBuffer(
    context,
    CL_MEM_READ_ONLY,
    SIZE * sizeof(int),
    NULL,
    NULL);

  cl_mem output = clCreateBuffer(
    context,
    CL_MEM_WRITE_ONLY,
    SIZE * sizeof(int),
    NULL,
    NULL);

  clEnqueueWriteBuffer(
    queue,
    input,
    CL_TRUE,
    0,
    SIZE * sizeof(int),
    data,
    0,
    NULL,
    NULL);

  size_t count = SIZE;

  clSetKernelArg(
    kernel,
    0,
    sizeof(cl_mem),
    &input);

  clSetKernelArg(
    kernel,
    1,
    sizeof(cl_mem),
    &output);

  cl_int err = clSetKernelArg(
    kernel,
    2,
    sizeof(size_t),
    &count);

  size_t group;

  clGetKernelWorkGroupInfo(
    kernel,
    device,
    CL_KERNEL_WORK_GROUP_SIZE,
    sizeof(size_t),
    &group,
    NULL);

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

  clEnqueueReadBuffer(
    queue,
    output,
    CL_TRUE,
    0,
    count * sizeof(int),
    result,
    0,
    NULL,
    NULL);

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  std::cout << std::endl << std::endl;
  for (size_t i = 0; i < SIZE; ++i)
    std::cout << data[i] << " ";
  std::cout << std::endl << std::endl;
  for (size_t i = 0; i < SIZE; ++i)
    std::cout << result[i] << " ";
  return 0;
}