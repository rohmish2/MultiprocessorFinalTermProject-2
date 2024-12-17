#include <iostream>
#include <cuda_runtime.h>

// Function to calculate the number of CUDA cores based on the compute capability
int getCudaCoresPerSM(int major, int minor) {
    // Mapping of compute capability to CUDA cores per multiprocessor
    if (major == 8) { // Ampere
        if (minor == 0) return 128; // A100
        if (minor == 6) return 128; // GA10x
    } else if (major == 7) { // Volta and Turing
        if (minor == 0) return 64;  // GV100
        if (minor == 5) return 64;  // T4, RTX 20xx
    } else if (major == 6) { // Pascal
        if (minor == 0 || minor == 1) return 64;  // P100
        if (minor == 2) return 128;               // Jetson TX2
    } else if (major == 5) { // Maxwell
        return 128;  // GTX 9xx
    } else if (major == 3) { // Kepler
        return 192;  // GTX 6xx/7xx
    }
    return -1; // Unknown compute capability
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found.\n";
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        int coresPerSM = getCudaCoresPerSM(deviceProp.major, deviceProp.minor);
        int totalCores = coresPerSM * deviceProp.multiProcessorCount;

        std::cout << "Device " << device << ": " << deviceProp.name << "\n";
        std::cout << "  CUDA Cores: " << totalCores << " (" << coresPerSM
                  << " per SM, " << deviceProp.multiProcessorCount << " SMs)\n";
        std::cout << "  Compute Capability: " << deviceProp.major << "."
                  << deviceProp.minor << "\n";
        std::cout << "  Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024)
                  << " MB\n\n";
    }

    return 0;
}

