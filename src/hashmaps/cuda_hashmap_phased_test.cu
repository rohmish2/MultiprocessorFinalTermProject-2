#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <algorithm>  
#include "cuda_hashmap.cuh"

struct TestResult {
    int block_size;
    int grid_size;
    double insert_time_ms;
    double search_time_ms;
    int failed_searches;
};

void __global__ insertKernel(uint32_t* keys, uint32_t* values,
                          const uint32_t* input_keys, const uint32_t* input_values,
                          uint32_t num_items, uint32_t array_size);

void __global__ findKernel(const uint32_t* keys, const uint32_t* values,
                        const uint32_t* search_keys, bool* results,
                        uint32_t num_items, uint32_t array_size);

TestResult runConfigTest(int block_size, const uint32_t NUM_ITEMS) {
     TestResult result;
    result.block_size = block_size;
    result.grid_size = (NUM_ITEMS + block_size - 1) / block_size;
    
    CUDAHashMap hashmap;
    uint32_t array_size = hashmap.getArraySize();
    
    std::vector<uint32_t> h_input_keys(NUM_ITEMS);
    std::vector<uint32_t> h_input_values(NUM_ITEMS);
    for (uint32_t i = 0; i < NUM_ITEMS; i++) {
        h_input_keys[i] = i + 1;
        h_input_values[i] = i + 1;
    }

    uint32_t *d_input_keys = nullptr;
    uint32_t *d_input_values = nullptr;
    bool *d_results = nullptr;
    
    cudaMalloc((void**)&d_input_keys, NUM_ITEMS * sizeof(uint32_t));
    cudaMalloc((void**)&d_input_values, NUM_ITEMS * sizeof(uint32_t));
    cudaMalloc((void**)&d_results, NUM_ITEMS * sizeof(bool));


    cudaMemcpy(d_input_keys, h_input_keys.data(), NUM_ITEMS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_values, h_input_values.data(), NUM_ITEMS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    std::cout << std::setw(8) << block_size << " threads/block: "
              << "Inserting " << NUM_ITEMS << " items... ";
    std::cout.flush();

    auto start = std::chrono::high_resolution_clock::now();
    
    insertKernel<<<result.grid_size, block_size>>>(
        hashmap.getKeys(), hashmap.getValues(),
        d_input_keys, d_input_values,
        NUM_ITEMS, array_size
    );
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    result.insert_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Done in " << std::fixed << std::setprecision(2) 
              << result.insert_time_ms << " ms\n";

    // Small delay between phases
    cudaDeviceSynchronize();

    std::cout << std::setw(8) << " " << "Searching for " << NUM_ITEMS << " items... ";
    std::cout.flush();

    start = std::chrono::high_resolution_clock::now();
    
    findKernel<<<result.grid_size, block_size>>>(
        hashmap.getKeys(), hashmap.getValues(),
        d_input_keys, d_results,
        NUM_ITEMS, array_size
    );
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    result.search_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Done in " << std::fixed << std::setprecision(2) 
              << result.search_time_ms << " ms\n";

   
std::vector<uint8_t> h_results(NUM_ITEMS);
cudaMemcpy(h_results.data(), d_results, NUM_ITEMS * sizeof(uint8_t), cudaMemcpyDeviceToHost);

result.failed_searches = 0;
for (uint32_t i = 0; i < NUM_ITEMS; i++) {
    if (h_results[i] == 0) {  
        result.failed_searches++;
    }
}

    // Cleanup
    cudaFree(d_input_keys);
    cudaFree(d_input_values);
    cudaFree(d_results);

    return result;
}

TestResult findBestResult(const std::vector<TestResult>& results) {
    TestResult best = results[0];
    for (const auto& result : results) {
        if ((result.insert_time_ms + result.search_time_ms) < 
            (best.insert_time_ms + best.search_time_ms)) {
            best = result;
        }
    }
    return best;
}

void printSummary(const std::vector<TestResult>& results, uint32_t num_items) {
    std::cout << "\nSummary for " << num_items << " items:\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(12) << "Block Size" 
              << std::setw(12) << "Grid Size"
              << std::setw(20) << "Insert Time (ms)"
              << std::setw(20) << "Search Time (ms)"
              << std::setw(20) << "Total Time (ms)"
              << std::setw(16) << "Throughput*"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    TestResult best_result = findBestResult(results); 

    for (const auto& result : results) {
        double total_time = result.insert_time_ms + result.search_time_ms;
        double throughput = (num_items * 2) / total_time * 1000; 
        bool is_best = (result.block_size == best_result.block_size);
        
        std::cout << std::setw(12) << result.block_size
                  << std::setw(12) << result.grid_size
                  << std::setw(20) << std::fixed << std::setprecision(2) << result.insert_time_ms
                  << std::setw(20) << result.search_time_ms
                  << std::setw(20) << total_time
                  << std::setw(16) << std::scientific << std::setprecision(2) << throughput
                  << (is_best ? " (Best)" : "")
                  << "\n";
        
        if (result.failed_searches > 0) {
            std::cout << "WARNING: " << result.failed_searches 
                      << " failed searches with block size " << result.block_size << "\n";
        }
    }
    std::cout << "\n* Throughput = total operations per second (insert + find)\n";
}

int main() {
    std::vector<uint32_t> problem_sizes = {1000000, 2000000, 4000000};
    
    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\nRunning on GPU: " << prop.name << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << "\n\n";

    for (uint32_t num_items : problem_sizes) {
        std::cout << "\nTesting with " << num_items << " items\n";
        std::cout << std::string(50, '=') << "\n";
        
        std::vector<TestResult> results;
        for (int block_size : block_sizes) {
            results.push_back(runConfigTest(block_size, num_items));
            std::cout << std::string(50, '-') << "\n";
        }
        
        printSummary(results, num_items);
    }

    return 0;
}