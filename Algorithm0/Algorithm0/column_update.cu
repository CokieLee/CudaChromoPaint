
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void ColumnUpdateKernel(int* colReads, int numHaps,
    double probTransition, double probError, int* prevStateProbs, int* currStateProbs)
{
    int i = threadIdx.x;
    for (int hap = 0; hap < numHaps; hap++) {
        if (hap = i) {
            currStateProbs[i] = (1 - probTransition) * probError;
        }
        else {
            currStateProbs[i] = probTransition * probError;
        }
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t KernelTime(int* colReads, int numHaps,
    double probTransition, double probError, int* prevStateProbs, int* currStateProbs, double* timeSec)
{
    int* dev_reads = 0;
    int* dev_prevStateProbs = 0;
    int* dev_currStateProbs = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_reads, numHaps * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_prevStateProbs, numHaps * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_currStateProbs, numHaps * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_reads, colReads, numHaps * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_prevStateProbs, prevStateProbs, numHaps * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    ColumnUpdateKernel << <1, numHaps >> > (dev_reads, numHaps, probTransition, probError, dev_prevStateProbs, dev_currStateProbs);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(currStateProbs, dev_currStateProbs, numHaps * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_reads);
    cudaFree(dev_prevStateProbs);
    cudaFree(dev_currStateProbs);

    return cudaStatus;
}

int main()
{
    const int minNumPeep = 10;
    const int maxNumPeep = 10'000;
    const int numTests = 300;

    const int seed = 4;

    const string outFile = "time.log";

    const int colSize = 5;
    const int a[colSize];
    for () {

    }

}
