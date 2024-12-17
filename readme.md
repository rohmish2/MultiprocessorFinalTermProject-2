login to tinker cliffs hpc cluster 
queue Implementation benchmark run (in tinker cliff)
g++ -fopenmp -std=c++17 -O3 -march=native queue_test_omp.cpp -I./MultiprocessorFinalTermProject/src/queues/queue_test_omp.cpp -o queue_test_omp 
srun --pty      --account=personal      --cpus-per-task=32      --time=01:00:00    ./queue_test_omp
hashmap  Implementation benchmark 
g++ -fopenmp -std=c++11 -I./libcuckoo  ./MultiprocessorFinalTermProject/src/hashmaps/hashmap_benchmark.cpp  -o m
g++ -fopenmp -std=c++11 -I./libcuckoo  ./MultiprocessorFinalTermProject/src/hashmaps/IOBenchmark.cpp  -o hashmap_benchmark

srun --pty      --account=personal      --cpus-per-task=32      --time=01:00:00    ./hashmap_benchmark
srun --pty      --account=personal      --cpus-per-task=32      --time=01:00:00    ./m


Cuda (login to inferarc clusters)
 module load CUDA/11.7.0

queue  
nvcc -std=c++11 ./MultiprocessorFinalTermProject/src/queues/main.cu -o m 

hashmap 
nvcc -std=c++11 ./MultiprocessorFinalTermProject/src/hashmaps/main.cu -o m 

to run :- 
./m 