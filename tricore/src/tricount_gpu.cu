#include "tricount.h"

#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "util.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native double atomicAdd is supported, do nothing!
#else
// Polyfill for older architectures that CMake might try to build
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

struct UndirEdge {
    uint32_t neighbor;
    uint32_t edge_id;
    bool operator<(const UndirEdge& other) const {
        return neighbor < other.neighbor;
    }
};

// Option A: The CPU Builder for Bidirectional Edge Mapping
void build_undirected_csr_cpu(
    const uint32_t* dag_edge_m,
    const uint32_t* dag_adj_m,
    uint32_t edge_count,
    uint32_t node_count,
    uint32_t** out_undir_node_index,
    uint32_t** out_undir_adj,
    uint32_t** out_undir_edge_id
) {
    printf(">>> CPU: Building Undirected Mapping CSR...\n");
    std::vector<std::vector<UndirEdge>> adj_list(node_count);

    // 1. Map both forward AND reverse edges, tied to the exact DAG Edge ID
    for (uint32_t i = 0; i < edge_count; i++) {
        uint32_t u = dag_edge_m[i];
        uint32_t v = dag_adj_m[i];
        adj_list[u].push_back({v, i}); // Forward edge
        adj_list[v].push_back({u, i}); // The missing reverse edge!
    }

    // 2. Allocate the host arrays (2x size because undirected)
    *out_undir_node_index = (uint32_t*)malloc((node_count + 1) * sizeof(uint32_t));
    *out_undir_adj = (uint32_t*)malloc(2 * edge_count * sizeof(uint32_t));
    *out_undir_edge_id = (uint32_t*)malloc(2 * edge_count * sizeof(uint32_t));

    // 3. Flatten and sort into strict CSR format
    uint32_t current_offset = 0;
    for (uint32_t i = 0; i < node_count; i++) {
        (*out_undir_node_index)[i] = current_offset;
        
        // Sorting is MANDATORY so the GPU 2-pointer intersection doesn't break
        std::sort(adj_list[i].begin(), adj_list[i].end());
        
        for (const auto& edge : adj_list[i]) {
            (*out_undir_adj)[current_offset] = edge.neighbor;
            (*out_undir_edge_id)[current_offset] = edge.edge_id;
            current_offset++;
        }
    }
    (*out_undir_node_index)[node_count] = current_offset;
    printf(">>> CPU: Undirected Mapping Built Successfully.\n");
}

#define CUDA_TRY(call)                                                          \
  do {                                                                          \
    cudaError_t const status = (call);                                          \
    if (cudaSuccess != status) {                                                \
      log_error("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                           \
  } while (0)

const int numBlocks = 65536;
const int BLOCKSIZE = 512;//1024;

uint64_t gpu_mem;

uint64_t init_gpu() {
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 512 * 1024 * 1024);
    cudaDeviceProp deviceProp{};
    CUDA_TRY(cudaGetDeviceProperties(&deviceProp, 0));
    gpu_mem = deviceProp.totalGlobalMem;
    log_info("numBlocks: %d  BLOCKSIZE: %d", numBlocks, BLOCKSIZE);
    return gpu_mem;
}

__global__ void all_degree_kernel(const uint64_t *edges, uint64_t edge_count, uint32_t *deg) {
    uint32_t blockSize = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (uint64_t i = tid; i < edge_count; i += blockSize) {
        uint64_t edge = edges[i];
        auto first = FIRST(edge);
        auto second = SECOND(edge);
        atomicAdd(deg + first, 1);
        atomicAdd(deg + second, 1);
    }
}

void cal_degree(const uint64_t *edges, uint64_t edge_count, uint32_t *deg, uint32_t node_count) {
    uint64_t use_mem = node_count * sizeof(uint32_t) + 1024 * 1024 * 256;
    uint64_t edge_block = (gpu_mem - use_mem) / sizeof(uint64_t);
    uint64_t split_num = edge_count / edge_block + 1;
    edge_block = edge_count / split_num;
    uint64_t *dev_edges;
    uint32_t *dev_deg;
    CUDA_TRY(cudaMalloc((void **) &dev_edges, edge_block * sizeof(uint64_t)));
    CUDA_TRY(cudaMalloc((void **) &dev_deg, node_count * sizeof(uint32_t)));
    for (uint64_t i = 0; i < edge_count; i += edge_block) {
        uint64_t copy_size = min(edge_count - i, edge_block);
        CUDA_TRY(cudaMemcpy(dev_edges, edges + i, copy_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
        all_degree_kernel<<<numBlocks, BLOCKSIZE>>> (dev_edges, copy_size, dev_deg);
        CUDA_TRY(cudaDeviceSynchronize());
    }
    CUDA_TRY(cudaMemcpy(deg, dev_deg, node_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaFree(dev_edges));
    CUDA_TRY(cudaFree(dev_deg));
}

__global__ void redirect_edges_kernel(uint64_t *edges, uint64_t edge_count, const uint32_t *deg) {
    uint32_t blockSize = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (uint64_t i = tid; i < edge_count; i += blockSize) {
        uint64_t edge = edges[i];
        auto first = FIRST(edge);
        auto second = SECOND(edge);
        if (deg[first] > deg[second] || (deg[first] == deg[second] && first > second)) {
            edges[i] = MAKEEDGE(second, first);
        }
    }
}

void redirect_edges(uint64_t *edges, uint64_t edge_count, const uint32_t *deg, uint32_t node_count) {
    uint64_t use_mem = node_count * sizeof(uint32_t) + 1024 * 1024 * 256;
    uint64_t edge_block = (gpu_mem - use_mem) / sizeof(uint64_t);
    uint64_t split_num = edge_count / edge_block + 1;
    edge_block = edge_count / split_num;
    uint64_t *dev_edges;
    uint32_t *dev_deg;
    CUDA_TRY(cudaMalloc((void **) &dev_edges, edge_block * sizeof(uint64_t)));
    CUDA_TRY(cudaMalloc((void **) &dev_deg, node_count * sizeof(uint32_t)));
    CUDA_TRY(cudaMemcpy(dev_deg, deg, node_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    for (uint64_t i = 0; i < edge_count; i += edge_block) {
        uint64_t copy_size = min(edge_count - i, edge_block);
        CUDA_TRY(cudaMemcpy(dev_edges, edges + i, copy_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
        redirect_edges_kernel<<< numBlocks, BLOCKSIZE>>> (dev_edges, copy_size, dev_deg);
        CUDA_TRY(cudaDeviceSynchronize());
        CUDA_TRY(cudaMemcpy(edges + i, dev_edges, copy_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }
    CUDA_TRY(cudaFree(dev_edges));
    CUDA_TRY(cudaFree(dev_deg));
}

uint32_t cal_part_num(const uint64_t *edges, uint64_t edge_count, uint32_t node_count) {
    uint64_t part_num = 1;
    while (true) {
        uint64_t part_edge_count = edge_count / part_num + 1;
        uint64_t part_node_count = node_count / part_num + 1;
        uint64_t tri_use_mem = part_edge_count * sizeof(uint64_t) * 2 * 115 / 100 + (part_node_count + 1) * sizeof(uint32_t) * 2 + numBlocks * BLOCKSIZE * sizeof(uint64_t);
        if (tri_use_mem < gpu_mem) {
            break;
        }
        ++part_num;
    }
    return part_num;
}

__global__ void unzip_edges_kernel(const uint64_t *edges, uint64_t edge_count, uint32_t *edges_first, uint32_t *edges_second) {
    auto from = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;
    for (uint64_t i = from; i < edge_count; i += step) {
        uint64_t tmp = edges[i];
        edges_first[i] = FIRST(tmp);
        edges_second[i] = SECOND(tmp);
    }
}

__global__ void node_index_construct_kernel(const uint32_t *edges_first, uint64_t edge_count, uint32_t *node_index, uint32_t node_count, uint32_t start_node) {
    auto from = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;
    for (uint64_t i = from; i <= edge_count; i += step) {
        int64_t prev = i > 0 ? (int64_t) (edges_first[i - 1] - start_node) : -1;
        int64_t next = i < edge_count ? (int64_t) (edges_first[i] - start_node) : node_count;
        for (int64_t j = prev + 1; j <= next; ++j) {
            node_index[j] = i;
        }
    }
}

struct is_self_loop : public thrust::unary_function<uint64_t, bool> {
    __host__ __device__
    bool operator()(uint64_t x) {
        return FIRST(x) == SECOND(x);
    }
};

void split(uint64_t *edges, uint64_t edge_count, uint64_t part_num, uint32_t **node_index, const uint32_t *node_split, uint64_t *adj_count) {
    uint64_t begin = 0;
    uint64_t end;
    uint64_t adj_count_all = 0;
    auto *adj = reinterpret_cast<uint32_t *>(edges);
    uint64_t *dev_edges;
    uint32_t *dev_edges_first;
    uint32_t *dev_edges_second;
    uint32_t *dev_node_index;

    for (uint64_t i = 0; i < part_num; i++) {
        log_info("split i: %lu start", i);
        uint32_t stop_node = node_split[i + 1];
        if (i == part_num - 1) {
            end = edge_count;
        } else {
            end = swap_if(edges + begin, edges + edge_count, [&](const uint64_t edge) {
                return FIRST(edge) < stop_node;
            }) - edges;
        }
        log_info("swap_if: %d start: %lu end: %lu", i, begin, end);
        adj_count[i] = end - begin;
        uint64_t copy_size = adj_count[i] * sizeof(uint64_t);
        CUDA_TRY(cudaMalloc((void **) &dev_edges, copy_size));
        CUDA_TRY(cudaMemcpy(dev_edges, edges + begin, copy_size, cudaMemcpyHostToDevice));
        thrust::device_ptr<uint64_t> dev_ptr(dev_edges);
        thrust::sort(dev_ptr, dev_ptr + adj_count[i]);
        adj_count[i] = thrust::remove_if(dev_ptr, dev_ptr + adj_count[i], is_self_loop()) - dev_ptr;
        adj_count[i] = thrust::unique(dev_ptr, dev_ptr + adj_count[i]) - dev_ptr;

        CUDA_TRY(cudaMalloc((void **) &dev_edges_first, adj_count[i] * sizeof(uint32_t)));
        CUDA_TRY(cudaMalloc((void **) &dev_edges_second, adj_count[i] * sizeof(uint32_t)));
        unzip_edges_kernel<<<numBlocks, BLOCKSIZE>>>(dev_edges, adj_count[i], dev_edges_first, dev_edges_second);
        CUDA_TRY(cudaPeekAtLastError());
        CUDA_TRY(cudaDeviceSynchronize());
        CUDA_TRY(cudaMemcpy(adj + adj_count_all, dev_edges_second, adj_count[i] * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t node_count = node_split[i + 1] - node_split[i] + 1;
        uint32_t start_node = node_split[i];
        CUDA_TRY(cudaMalloc((void **) &dev_node_index, (node_count + 1) * sizeof(uint32_t)));
        node_index_construct_kernel<<<numBlocks, BLOCKSIZE>>>(dev_edges_first, adj_count[i], dev_node_index, node_count, start_node);
        CUDA_TRY(cudaPeekAtLastError());
        CUDA_TRY(cudaDeviceSynchronize());
        node_index[i] = (uint32_t *) malloc((node_count + 1) * sizeof(uint32_t));
        CUDA_TRY(cudaMemcpy(node_index[i], dev_node_index, (node_count + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        adj_count_all += adj_count[i];

        CUDA_TRY(cudaFree(dev_edges));
        CUDA_TRY(cudaFree(dev_edges_first));
        CUDA_TRY(cudaFree(dev_edges_second));
        CUDA_TRY(cudaFree(dev_node_index));
        begin = end;
    }
    vector<uint64_t> adj_count_vec(adj_count, adj_count + part_num);
    adj_count[0] = 0;
    for (uint64_t i = 1; i <= part_num; i++) {
        adj_count[i] = adj_count[i - 1] + adj_count_vec[i - 1];
    }
}

__global__ void node_index_reconstruct_kernel(uint32_t *edges_first, const uint32_t *node_index, uint32_t node_count) {
    auto from = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;
    for (uint64_t i = from; i < node_count; i += step) {
        for (uint64_t j = node_index[i]; j < node_index[i + 1]; ++j) {
            edges_first[j] = i;
        }
    }
}

__global__ void warp_binary_kernel(const uint32_t* __restrict__ edge_m, const uint32_t* __restrict__ node_index_m, uint32_t edge_m_count, uint32_t* __restrict__ adj_m, uint32_t start_node_n, const uint32_t* __restrict__ node_index_n, uint32_t node_index_n_count, uint32_t* __restrict__ adj_n, uint64_t *results) {
    //phase 1, partition
    uint64_t count = 0;
    __shared__ uint32_t local[BLOCKSIZE];

    uint32_t i = threadIdx.x % 32;
    uint32_t p = threadIdx.x / 32;
    for (uint32_t tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32; tid < edge_m_count; tid += blockDim.x * gridDim.x / 32) {
        uint32_t node_m = edge_m[tid];
        uint32_t node_n = adj_m[tid];
        if (node_n < start_node_n || node_n >= start_node_n + node_index_n_count) {
            continue;
        }

        uint32_t degree_m = node_index_m[node_m + 1] - node_index_m[node_m];
        uint32_t degree_n = node_index_n[node_n + 1 - start_node_n] - node_index_n[node_n - start_node_n];
        uint32_t* a = adj_m + node_index_m[node_m];
        uint32_t* b = adj_n + node_index_n[node_n - start_node_n];
        if(degree_m < degree_n){
            uint32_t temp = degree_m;
            degree_m = degree_n;
            degree_n = temp;
            uint32_t *aa = a;
            a = b;
            b = aa;
        }

        //initial cache
        local[p * 32 + i] = a[i * degree_m / 32];
        __syncthreads();

        //search
        uint32_t j = i;
        while(j < degree_n){
            uint32_t X = b[j];
            uint32_t Y;
            //phase 1: cache
            int32_t bot = 0;
            int32_t top = 32;
            int32_t r;
            while(top > bot + 1){
                r = (top + bot) / 2;
                Y = local[p * 32 + r];
                if(X == Y){
                    count++;
                    bot = top + 32;
                }
                if(X < Y){
                    top = r;
                }
                if(X > Y){
                    bot = r;
                }
            }
            //phase 2
            bot = bot * degree_m / 32;
            top = top * degree_m / 32 - 1;
            while(top >= bot){
                r = (top + bot) / 2;
                Y = a[r];
                if(X == Y){
                    count++;
                }
                if(X <= Y){
                    top = r - 1;
                }
                if(X >= Y){
                    bot = r + 1;
                }
            }
            j += 32;
        }
    // 1. Squeeze the partial counts
        uint64_t warp_sum = count;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        // 2. Thread 0 saves the exact edge weight directly into the VRAM array!
        if (i == 0) {
            results[tid] = warp_sum; 
        }
        
        // 3. CRITICAL FIX: Reset the count to 0 for the next edge in the loop!
        count = 0;
        __syncthreads();
    }
    // results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}

// 1. EXACT FRACTIONAL INITIALIZATION
__global__ void init_exact_weights_kernel(
    const uint32_t* edge_m, const uint32_t* adj_m,
    const uint32_t* undir_node_index, const uint32_t* undir_adj,
    double* exact_tri_counts, uint32_t edge_count // <-- NOW DOUBLE
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;

    uint32_t u = edge_m[i];
    uint32_t v = adj_m[i];

    uint32_t ptr_u = undir_node_index[u];
    uint32_t end_u = undir_node_index[u+1];
    uint32_t ptr_v = undir_node_index[v];
    uint32_t end_v = undir_node_index[v+1];

    double deg_u = (double)(end_u - ptr_u);
    double deg_v = (double)(end_v - ptr_v);

    double weight_sum = 0.0;
    while (ptr_u < end_u && ptr_v < end_v) {
        uint32_t w_u = undir_adj[ptr_u];
        uint32_t w_v = undir_adj[ptr_v];

        if (w_u == w_v) {
            double deg_w = (double)(undir_node_index[w_u+1] - undir_node_index[w_u]);
            weight_sum += 1.0 / (deg_u * deg_v * deg_w); // CPU's exact weight!
            ptr_u++; ptr_v++;
        } 
        else if (w_u < w_v) ptr_u++;
        else ptr_v++;
    }
    exact_tri_counts[i] = weight_sum;
}

// 2. EXACT FRACTIONAL THRESHOLD CHECKER
__global__ void checker_kernel(
    const uint32_t* edge_m, const uint32_t* adj_m,
    const uint32_t* node_index, const double* tri_counts, // <-- NOW DOUBLE
    bool* is_active, bool* just_deleted,
    double eps, uint32_t edge_count, bool* d_changed
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    if (!is_active[i]) return;

    uint32_t u = edge_m[i];
    uint32_t v = adj_m[i];

    double deg_u = (double)(node_index[u+1] - node_index[u]);
    double deg_v = (double)(node_index[v+1] - node_index[v]);
    
    // THE CPU EXACT JACCARD THRESHOLD
    double edgeWt = 1.0 / (deg_u * deg_v);
    double threshold = eps * edgeWt;

    // We keep a microscopic armor just to prevent IEEE float mismatches
    if (tri_counts[i] < threshold - 1e-9) {
        is_active[i] = false;
        just_deleted[i] = true;
        *d_changed = true;
    }
}

// 3. EXACT FRACTIONAL UPDATER
__global__ void updater_kernel(
    const uint32_t* edge_m, const uint32_t* adj_m,
    const uint32_t* undir_node_index, const uint32_t* undir_adj,
    const uint32_t* undir_edge_id, double* tri_counts, // <-- NOW DOUBLE
    bool* is_active, bool* just_deleted, uint32_t edge_count
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    if (!just_deleted[i]) return;

    uint32_t u = edge_m[i];
    uint32_t v = adj_m[i];

    uint32_t ptr_u = undir_node_index[u];
    uint32_t end_u = undir_node_index[u+1];
    uint32_t ptr_v = undir_node_index[v];
    uint32_t end_v = undir_node_index[v+1];

    double deg_u = (double)(end_u - ptr_u);
    double deg_v = (double)(end_v - ptr_v);

    while (ptr_u < end_u && ptr_v < end_v) {
        uint32_t w_u = undir_adj[ptr_u];
        uint32_t w_v = undir_adj[ptr_v];

        if (w_u == w_v) {
            uint32_t id_uw = undir_edge_id[ptr_u]; 
            uint32_t id_vw = undir_edge_id[ptr_v]; 

            // Calculate the exact fractional weight of this dying triangle
            double deg_w = (double)(undir_node_index[w_u+1] - undir_node_index[w_u]);
            double decrement_weight = 1.0 / (deg_u * deg_v * deg_w);

            bool a_active = is_active[id_uw];
            bool a_just_dead = just_deleted[id_uw];
            bool b_active = is_active[id_vw];
            bool b_just_dead = just_deleted[id_vw];

            // Use atomicAdd to subtract the fraction!
            if (a_active) {
                if (b_active || (b_just_dead && i < id_vw)) {
                    atomicAdd(&tri_counts[id_uw], -decrement_weight);
                }
            }
            if (b_active) {
                if (a_active || (a_just_dead && i < id_uw)) {
                    atomicAdd(&tri_counts[id_vw], -decrement_weight);
                }
            }
            ptr_u++; ptr_v++;
        } 
        else if (w_u < w_v) ptr_u++;
        else ptr_v++;
    }
}




__global__ void warp_intersection_kernel(const uint32_t* __restrict__ edge_m, 
										 const uint32_t* __restrict__ node_index_m, 
										 uint32_t edge_m_count, 
										 uint32_t* __restrict__ adj_m, 
										 uint32_t start_node_n, 
										 const uint32_t* __restrict__ node_index_n, 
										 uint32_t node_index_n_count, 
										 uint32_t* __restrict__ adj_n, 
										 uint64_t *results) {
    //phase 1, partition
    uint64_t count = 0;
    //__shared__ uint32_t local[BLOCKSIZE];

    //uint32_t i = threadIdx.x % 32;
    //uint32_t p = threadIdx.x / 32;
    for (uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < edge_m_count; tid += blockDim.x * gridDim.x) {
        uint32_t node_m = edge_m[tid];
        uint32_t node_n = adj_m[tid];
        if (node_n < start_node_n || node_n >= start_node_n + node_index_n_count) {
            continue;
        }

        uint32_t degree_m = node_index_m[node_m + 1] - node_index_m[node_m];
        uint32_t degree_n = node_index_n[node_n + 1 - start_node_n] - node_index_n[node_n - start_node_n];
        uint32_t* a = adj_m + node_index_m[node_m];
        uint32_t* b = adj_n + node_index_n[node_n - start_node_n];

        //initial cache
		int i = 0, j = 0;
		while (i < degree_m && j < degree_n) {
			if (a[i] == b[j]) {
				count ++;
				i ++;
				j ++;
			} else if (a[i] < b[j]) {
				i ++;
			} else {
				j ++;
			}
		}
        //search
    }
    results[blockDim.x * blockIdx.x + threadIdx.x] = count;
}


uint64_t tricount_gpu(uint64_t part_num, uint32_t *adj, const uint64_t *adj_count, const uint32_t *node_split, uint32_t **node_index) {
    uint32_t n_result = numBlocks * BLOCKSIZE;
    uint64_t all_sum = 0;
    uint32_t *node_index_m_dev;
    uint32_t *adj_m_dev;
    uint32_t *edge_m_dev;
    uint32_t *node_index_n_dev;
    uint32_t *adj_n_dev;
    uint32_t *edge_n_dev;
    uint64_t *dev_results;
    for (uint64_t m = 0; m < part_num; m++) {
        uint32_t start_node_m = node_split[m];
        uint32_t *node_index_m = node_index[m];
        uint32_t node_index_m_count = node_split[m + 1] - node_split[m];
        if (node_index_m_count == 0) {
            continue;
        }
        uint64_t start_adj_m = adj_count[m];
        uint32_t *adj_m = adj + start_adj_m;
        uint32_t adj_count_m = adj_count[m + 1] - adj_count[m];
        CUDA_TRY(cudaMalloc((void **) &node_index_m_dev, (node_index_m_count + 1) * sizeof(uint32_t)));
        CUDA_TRY(cudaMemcpy(node_index_m_dev, node_index_m, (node_index_m_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMalloc((void **) &adj_m_dev, adj_count_m * sizeof(uint32_t)));
        CUDA_TRY(cudaMemcpy(adj_m_dev, adj_m, adj_count_m * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMalloc((void **) &edge_m_dev, adj_count_m * sizeof(uint32_t)));
        node_index_reconstruct_kernel<<<numBlocks, BLOCKSIZE>>>(edge_m_dev, node_index_m_dev, node_index_m_count);
        CUDA_TRY(cudaDeviceSynchronize());
        for (uint64_t n = m; n < part_num; n++) {
            uint32_t start_node_n = node_split[n];
            uint32_t *node_index_n = node_index[n];
            uint32_t node_index_n_count = node_split[n + 1] - node_split[n];
            uint64_t start_adj_n = adj_count[n];
            uint32_t *adj_n = adj + start_adj_n;
            uint32_t adj_count_n = adj_count[n + 1] - adj_count[n];
            CUDA_TRY(cudaMalloc((void **) &node_index_n_dev, (node_index_n_count + 1) * sizeof(uint32_t)));
            CUDA_TRY(cudaMemcpy(node_index_n_dev, node_index_n, (node_index_n_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_TRY(cudaMalloc((void **) &adj_n_dev, adj_count_n * sizeof(uint32_t)));
            CUDA_TRY(cudaMemcpy(adj_n_dev, adj_n, adj_count_n * sizeof(uint32_t), cudaMemcpyHostToDevice));

            CUDA_TRY(cudaMalloc((void **) &dev_results, adj_count_m * sizeof(uint64_t)));
//            log_info("tricount_gpu_edge_kernel start");
            warp_binary_kernel<<<numBlocks, BLOCKSIZE>>>(edge_m_dev, node_index_m_dev, adj_count_m, adj_m_dev, start_node_n, node_index_n_dev, node_index_n_count, adj_n_dev, dev_results);
            CUDA_TRY(cudaDeviceSynchronize());
//            log_info("tricount_gpu_edge_kernel end");

            // --- BUG FIX: Reduce only up to adj_count_m to prevent segfault! ---
            thrust::device_ptr<uint64_t> ptr(dev_results);
            uint64_t sum = thrust::reduce(ptr, ptr + adj_count_m); 
            log_info("m: %d n: %d sum: %lu", m, n, sum);
            all_sum += sum;

            // ====================================================================
            // >>> HETEROGENEOUS HANDOFF: LAUNCH NATIVE GPU PEELING <<<
            // ====================================================================
            if (m == 0 && n == 0) { // Ensures this fires on the primary graph
                double eps = 0.1;
                // Launch the wrapper we built earlier!
                execute_gpu_peeling(edge_m_dev, adj_m_dev, node_index_m_dev, dev_results, adj_count_m, node_split[1], eps);
            }
            // ====================================================================

            if (m != n) {
                CUDA_TRY(cudaMalloc((void **) &edge_n_dev, adj_count_n * sizeof(uint32_t)));
            }
            CUDA_TRY(cudaFree(node_index_n_dev));
            CUDA_TRY(cudaFree(adj_n_dev));
            CUDA_TRY(cudaFree(dev_results));
        }
        CUDA_TRY(cudaFree(node_index_m_dev));
        CUDA_TRY(cudaFree(adj_m_dev));
        CUDA_TRY(cudaFree(edge_m_dev));
    }
    return all_sum;
}

// =========================================================================
// >>> DIAGNOSTIC: EXACT INTEGER TRIANGLE COUNTER <<<
// =========================================================================
__global__ void verify_integer_triangles_kernel(
    const uint32_t* edge_m, const uint32_t* adj_m,
    const uint32_t* undir_node_index, const uint32_t* undir_adj,
    const uint32_t* undir_edge_id, const bool* is_active, 
    uint32_t edge_count, unsigned long long* total_integer_triangles
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    
    // If this edge died during peeling, it cannot be part of a surviving triangle
    if (!is_active[i]) return; 

    uint32_t u = edge_m[i];
    uint32_t v = adj_m[i];

    uint32_t ptr_u = undir_node_index[u];
    uint32_t end_u = undir_node_index[u+1];
    uint32_t ptr_v = undir_node_index[v];
    uint32_t end_v = undir_node_index[v+1];

    unsigned long long local_count = 0;

    while (ptr_u < end_u && ptr_v < end_v) {
        uint32_t w_u = undir_adj[ptr_u];
        uint32_t w_v = undir_adj[ptr_v];

        if (w_u == w_v) {
            uint32_t id_uw = undir_edge_id[ptr_u];
            uint32_t id_vw = undir_edge_id[ptr_v];
            
            // A triangle only survives if ALL THREE of its edges survived the cascade!
            if (is_active[id_uw] && is_active[id_vw]) {
                local_count++;
            }
            ptr_u++; ptr_v++;
        } 
        else if (w_u < w_v) ptr_u++;
        else ptr_v++;
    }
    
    if (local_count > 0) {
        atomicAdd(total_integer_triangles, local_count);
    }
}

void execute_gpu_peeling(
    const uint32_t* dev_edge_m,
    const uint32_t* dev_adj_m,
    const uint32_t* dev_node_index,
    uint64_t* dev_tri_counts,
    uint32_t edge_count,
    uint32_t node_count,
    double eps
) {
    printf("\n>>> INITIATING PURE GPU PEELING LOOP <<<\n");
    
    // --- DEFINE GRID FIRST SO KERNELS CAN USE IT ---
    int threads = 512;
    int blocks = (edge_count + threads - 1) / threads;

    // --- OPTION A: CPU BUILDER TO GPU TRANSFER ---
    uint32_t *host_undir_node_index, *host_undir_adj, *host_undir_edge_id;
    
    // We need the DAG arrays back on the host to build the map
    uint32_t* host_dag_edge_m = (uint32_t*)malloc(edge_count * sizeof(uint32_t));
    uint32_t* host_dag_adj_m = (uint32_t*)malloc(edge_count * sizeof(uint32_t));
    CUDA_TRY(cudaMemcpy(host_dag_edge_m, dev_edge_m, edge_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(host_dag_adj_m, dev_adj_m, edge_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    build_undirected_csr_cpu(host_dag_edge_m, host_dag_adj_m, edge_count, node_count, 
                             &host_undir_node_index, &host_undir_adj, &host_undir_edge_id);

    // Allocate VRAM for the Undirected Arrays
    uint32_t *dev_undir_node_index, *dev_undir_adj, *dev_undir_edge_id;
    CUDA_TRY(cudaMalloc((void**)&dev_undir_node_index, (node_count + 1) * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc((void**)&dev_undir_adj, 2 * edge_count * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc((void**)&dev_undir_edge_id, 2 * edge_count * sizeof(uint32_t)));

    // Push across the PCIe bus
    CUDA_TRY(cudaMemcpy(dev_undir_node_index, host_undir_node_index, (node_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(dev_undir_adj, host_undir_adj, 2 * edge_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(dev_undir_edge_id, host_undir_edge_id, 2 * edge_count * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // --- INITIALIZE EXACT PER-EDGE WEIGHTS ---
    double* dev_exact_tri_counts;
    CUDA_TRY(cudaMalloc((void**)&dev_exact_tri_counts, edge_count * sizeof(double)));

    printf(">>> GPU: Initializing Exact Per-Edge Triangle Weights...\n");
    init_exact_weights_kernel<<<blocks, threads>>>(
        dev_edge_m, dev_adj_m, dev_undir_node_index, dev_undir_adj, dev_exact_tri_counts, edge_count
    );
    CUDA_TRY(cudaDeviceSynchronize());

    bool *dev_is_active, *dev_just_deleted, *dev_changed;
    CUDA_TRY(cudaMalloc((void**)&dev_is_active, edge_count * sizeof(bool)));
    CUDA_TRY(cudaMalloc((void**)&dev_just_deleted, edge_count * sizeof(bool)));
    CUDA_TRY(cudaMalloc((void**)&dev_changed, sizeof(bool)));

    CUDA_TRY(cudaMemset(dev_is_active, 1, edge_count * sizeof(bool)));
    CUDA_TRY(cudaMemset(dev_just_deleted, 0, edge_count * sizeof(bool)));

    bool host_changed = true;
    int iteration = 0;

    while (host_changed) {
        iteration++;
        host_changed = false;
        
        CUDA_TRY(cudaMemcpy(dev_changed, &host_changed, sizeof(bool), cudaMemcpyHostToDevice));

        checker_kernel<<<blocks, threads>>>(
            dev_edge_m, dev_adj_m, dev_undir_node_index, dev_exact_tri_counts, dev_is_active, dev_just_deleted, eps, edge_count, dev_changed
        );
        CUDA_TRY(cudaDeviceSynchronize());

        CUDA_TRY(cudaMemcpy(&host_changed, dev_changed, sizeof(bool), cudaMemcpyDeviceToHost));

        if (host_changed) {
            updater_kernel<<<blocks, threads>>>(
                dev_edge_m, dev_adj_m, dev_undir_node_index, dev_undir_adj, dev_undir_edge_id, dev_exact_tri_counts, dev_is_active, dev_just_deleted, edge_count
            );
            CUDA_TRY(cudaDeviceSynchronize());
            
            // >>> THE FIX: Safely wipe the flags globally after all warps have finished! <<<
            CUDA_TRY(cudaMemset(dev_just_deleted, 0, edge_count * sizeof(bool)));

            printf("Iteration %d: Masked edges virtually deleted. Cascading updates triggered.\n", iteration);
        } else {
            printf("Iteration %d: Graph stabilized. Peeling complete.\n", iteration);
        }
    }
    
    thrust::device_ptr<bool> mask_ptr(dev_is_active);
    int surviving_edges = thrust::count(mask_ptr, mask_ptr + edge_count, true);
    int deleted_edges = edge_count - surviving_edges;
    
    printf("\n--- PEELING VERIFICATION STATS ---\n");
    printf("Original Edges: %u\n", edge_count);
    printf("Deleted Edges:  %d\n", deleted_edges);
    printf("Surviving Edges: %d\n", surviving_edges);
    
//     // ====================================================================
//     // >>> VERIFICATION: EXACT SURVIVING FRACTIONAL WEIGHT <<<
//     // ====================================================================
//     bool* host_is_active = (bool*)malloc(edge_count * sizeof(bool));
    
//     // 1. MUST BE ALLOCATED AS DOUBLE!
//     double* host_tri_counts = (double*)malloc(edge_count * sizeof(double)); 
    
//     CUDA_TRY(cudaMemcpy(host_is_active, dev_is_active, edge_count * sizeof(bool), cudaMemcpyDeviceToHost));
//     CUDA_TRY(cudaMemcpy(host_tri_counts, dev_exact_tri_counts, edge_count * sizeof(double), cudaMemcpyDeviceToHost));

//     // 2. MUST BE SUMMED AS A DOUBLE!
//     double surviving_weight_sum = 0.0; 
//     for (uint32_t i = 0; i < edge_count; i++) {
//         if (host_is_active[i]) {
//             surviving_weight_sum += host_tri_counts[i];
//         }
//     }
    
//     // Each triangle's fractional weight is shared by 3 edges
//     printf("Total fractional triangle weight, post cleaning = %f\n", surviving_weight_sum / 3.0);
//     printf("----------------------------------\n\n");

//     free(host_is_active);
//     free(host_tri_counts);
//     free(host_dag_edge_m);
//     free(host_dag_adj_m);

//     CUDA_TRY(cudaFree(dev_is_active));
//     CUDA_TRY(cudaFree(dev_just_deleted));
//     CUDA_TRY(cudaFree(dev_changed));
//     CUDA_TRY(cudaFree(dev_exact_tri_counts));
//     CUDA_TRY(cudaFree(dev_undir_node_index));
//     CUDA_TRY(cudaFree(dev_undir_adj));
//     CUDA_TRY(cudaFree(dev_undir_edge_id));
// }

// ====================================================================
    // >>> VERIFICATION: EXACT SURVIVING INTEGER TRIANGLES <<<
    // ====================================================================
    unsigned long long* dev_total_integer_triangles;
    CUDA_TRY(cudaMalloc((void**)&dev_total_integer_triangles, sizeof(unsigned long long)));
    CUDA_TRY(cudaMemset(dev_total_integer_triangles, 0, sizeof(unsigned long long)));

    printf(">>> GPU: Running Diagnostic Integer Triangle Scan...\n");
    verify_integer_triangles_kernel<<<blocks, threads>>>(
        dev_edge_m, dev_adj_m, dev_undir_node_index, dev_undir_adj, dev_undir_edge_id, dev_is_active, edge_count, dev_total_integer_triangles
    );
    CUDA_TRY(cudaDeviceSynchronize());

    unsigned long long host_total_integer_triangles = 0;
    CUDA_TRY(cudaMemcpy(&host_total_integer_triangles, dev_total_integer_triangles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Because we iterate over all edges, every surviving triangle is counted exactly 3 times (once by each of its 3 edges).
    unsigned long long exact_surviving_triangles = host_total_integer_triangles / 3;

    printf("Total integer triangle count, post cleaning = %llu\n", exact_surviving_triangles);
    printf("----------------------------------\n\n");

    CUDA_TRY(cudaFree(dev_total_integer_triangles));
    
    // --- FREE REMAINING VRAM ---
    free(host_dag_edge_m);
    free(host_dag_adj_m);
    CUDA_TRY(cudaFree(dev_is_active));
    CUDA_TRY(cudaFree(dev_just_deleted));
    CUDA_TRY(cudaFree(dev_changed));
    CUDA_TRY(cudaFree(dev_exact_tri_counts));
    CUDA_TRY(cudaFree(dev_undir_node_index));
    CUDA_TRY(cudaFree(dev_undir_adj));
    CUDA_TRY(cudaFree(dev_undir_edge_id));
}