#include <nvrtc.h>
#include <cuda.h>
#include <nvJitLink.h>
#include <iostream>

void foo();

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define NVJITLINK_SAFE_CALL(h,x)                                  \
  do {                                                            \
    nvJitLinkResult result = x;                                   \
    if (result != NVJITLINK_SUCCESS) {                            \
      std::cerr << "\nerror: " #x " failed with error "           \
                << result << '\n';                                \
      size_t lsize;                                               \
      result = nvJitLinkGetErrorLogSize(h, &lsize);               \
      if (result == NVJITLINK_SUCCESS && lsize > 0) {             \
        char *log = (char*)malloc(lsize);                         \
    result = nvJitLinkGetErrorLog(h, log);                        \
    if (result == NVJITLINK_SUCCESS) {                            \
      std::cerr << "error: " << log << '\n';                      \
      free(log);                                                  \
    }                                                             \
      }                                                           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

const char* saxpy = "                                           \n\
extern __device__ float compute(float a, float x, float y);     \n\
                                                                \n\
extern \"C\" __global__                                         \n\
void saxpy(float a, float *x, float *y, float *out, size_t n)   \n\
{                                                               \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
  if (tid < n) {                                                \n\
    out[tid] = compute(a, x[tid], y[tid]);                      \n\
  }                                                             \n\
}                                                               \n";

int main(int argc, char* argv[])
{
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

    int major = 0;
    int minor = 0;
    CUDA_SAFE_CALL(cuDeviceGetAttribute(&major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CUDA_SAFE_CALL(cuDeviceGetAttribute(&minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    int arch = major * 10 + minor;
    char smbuf[16];
    sprintf(smbuf, "-arch=sm_%d", arch);

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, saxpy, "saxpy", 0, NULL, NULL));

    const char* opts[] = { smbuf };
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);
    size_t PTXSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &PTXSize));
    char* PTX = new char[PTXSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, PTX));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    nvJitLinkHandle handle;
    NVJITLINK_SAFE_CALL(handle, nvJitLinkCreate(&handle, 1, opts));
    NVJITLINK_SAFE_CALL(handle, nvJitLinkAddFile(handle, NVJITLINK_INPUT_OBJECT, "offline.o"));
    NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, (void*)PTX, PTXSize, "online"));

    NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));
    size_t cubinSize;
    NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
    void* cubin = malloc(cubinSize);
    NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubin(handle, cubin));
    NVJITLINK_SAFE_CALL(handle, nvJitLinkDestroy(&handle));
    CUDA_SAFE_CALL(cuModuleLoadData(&module, cubin));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "saxpy"));

    size_t n = 10;
    size_t bufferSize = n * sizeof(float);
    float a = 5.1f;
    float* hX = new float[n], * hY = new float[n], * hOut = new float[n];
    for (size_t i = 0; i < n; ++i) {
        hX[i] = static_cast<float>(i);
        hY[i] = static_cast<float>(i * 2);
    }
    CUdeviceptr dX, dY, dOut;
    CUDA_SAFE_CALL(cuMemAlloc(&dX, bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dY, bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));
    void* args[] = { &a, &dX, &dY, &dOut, &n };
    CUDA_SAFE_CALL(
        cuLaunchKernel(kernel,
            1, 1, 1,             // grid dim
            n, 1, 1,             // block dim
            0, NULL,             // shared mem and stream
            args, 0));           // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());
    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));

    for (size_t i = 0; i < n; ++i) {
        std::cout << a << " * " << hX[i] << " + " << hY[i]
            << " = " << hOut[i] << '\n';
    }

    CUDA_SAFE_CALL(cuMemFree(dX));
    CUDA_SAFE_CALL(cuMemFree(dY));
    CUDA_SAFE_CALL(cuMemFree(dOut));
    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuCtxDestroy(context));
    free(cubin);
    delete[] hX;
    delete[] hY;
    delete[] hOut;
    delete[] PTX;

    foo();

    return 0;
}
