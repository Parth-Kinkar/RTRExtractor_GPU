#include "tricount.h"

// =========================================================================
// >>> 1. THRUST AND CUDA INCLUDES <<<
// =========================================================================
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/distance.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>

#include "util.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native support exists for sm_60 and above, no polyfill needed.
#else
// Polyfill for older architectures
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif



struct Profiler {
    double vram_alloc_time = 0;
    double pcie_transfer_time = 0;
    double gpu_compute_time = 0;
    double cpu_compute_time = 0;
};

struct UndirEdge {
    uint32_t neighbor;
    uint32_t edge_id;
    bool operator<(const UndirEdge& other) const { return neighbor < other.neighbor; }
};

void build_undirected_csr_cpu(
    const uint32_t* dag_edge_m, const uint32_t* dag_adj_m, uint32_t edge_count, uint32_t node_count,
    uint32_t** out_undir_node_index, uint32_t** out_undir_adj, uint32_t** out_undir_edge_id
) {
    printf(">>> CPU: Building Undirected Mapping CSR...\n");
    std::vector<std::vector<UndirEdge>> adj_list(node_count);
    for (uint32_t i = 0; i < edge_count; i++) {
        uint32_t u = dag_edge_m[i]; uint32_t v = dag_adj_m[i];
        adj_list[u].push_back({v, i}); adj_list[v].push_back({u, i});
    }
    *out_undir_node_index = (uint32_t*)malloc((node_count + 1) * sizeof(uint32_t));
    *out_undir_adj = (uint32_t*)malloc(2 * edge_count * sizeof(uint32_t));
    *out_undir_edge_id = (uint32_t*)malloc(2 * edge_count * sizeof(uint32_t));
    uint32_t current_offset = 0;
    for (uint32_t i = 0; i < node_count; i++) {
        (*out_undir_node_index)[i] = current_offset;
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

#define CUDA_TRY(call) \
  do { cudaError_t const status = (call); \
    if (cudaSuccess != status) { log_error("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__); } \
  } while (0)

const int numBlocks = 65536;
const int BLOCKSIZE = 512;
uint64_t gpu_mem;

uint64_t init_gpu() {
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 512 * 1024 * 1024);
    cudaDeviceProp deviceProp{};
    CUDA_TRY(cudaGetDeviceProperties(&deviceProp, 0));
    gpu_mem = deviceProp.totalGlobalMem;
    return gpu_mem;
}

__global__ void all_degree_kernel(const uint64_t *edges, uint64_t edge_count, uint32_t *deg) {
    uint32_t blockSize = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (uint64_t i = tid; i < edge_count; i += blockSize) {
        uint64_t edge = edges[i];
        atomicAdd(deg + FIRST(edge), 1);
        atomicAdd(deg + SECOND(edge), 1);
    }
}

void cal_degree(const uint64_t *edges, uint64_t edge_count, uint32_t *deg, uint32_t node_count) {
    uint64_t use_mem = node_count * sizeof(uint32_t) + 1024 * 1024 * 256;
    uint64_t edge_block = (gpu_mem - use_mem) / sizeof(uint64_t);
    uint64_t split_num = edge_count / edge_block + 1;
    edge_block = edge_count / split_num;
    uint64_t *dev_edges; uint32_t *dev_deg;
    CUDA_TRY(cudaMalloc((void **) &dev_edges, edge_block * sizeof(uint64_t)));
    CUDA_TRY(cudaMalloc((void **) &dev_deg, node_count * sizeof(uint32_t)));
    for (uint64_t i = 0; i < edge_count; i += edge_block) {
        uint64_t copy_size = min(edge_count - i, edge_block);
        CUDA_TRY(cudaMemcpy(dev_edges, edges + i, copy_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
        all_degree_kernel<<<numBlocks, BLOCKSIZE>>> (dev_edges, copy_size, dev_deg);
        CUDA_TRY(cudaDeviceSynchronize());
    }
    CUDA_TRY(cudaMemcpy(deg, dev_deg, node_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaFree(dev_edges)); CUDA_TRY(cudaFree(dev_deg));
}

__global__ void redirect_edges_kernel(uint64_t *edges, uint64_t edge_count, const uint32_t *deg) {
    uint32_t blockSize = blockDim.x * gridDim.x;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (uint64_t i = tid; i < edge_count; i += blockSize) {
        uint64_t edge = edges[i];
        auto first = FIRST(edge); auto second = SECOND(edge);
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
    uint64_t *dev_edges; uint32_t *dev_deg;
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
    CUDA_TRY(cudaFree(dev_edges)); CUDA_TRY(cudaFree(dev_deg));
}

uint32_t cal_part_num(const uint64_t *edges, uint64_t edge_count, uint32_t node_count) {
    uint64_t part_num = 1;
    while (true) {
        uint64_t part_edge_count = edge_count / part_num + 1;
        uint64_t part_node_count = node_count / part_num + 1;
        uint64_t tri_use_mem = part_edge_count * sizeof(uint64_t) * 2 * 115 / 100 + (part_node_count + 1) * sizeof(uint32_t) * 2 + numBlocks * BLOCKSIZE * sizeof(uint64_t);
        if (tri_use_mem < gpu_mem) break;
        ++part_num;
    }
    return part_num;
}

__global__ void unzip_edges_kernel(const uint64_t *edges, uint64_t edge_count, uint32_t *edges_first, uint32_t *edges_second) {
    auto from = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;
    for (uint64_t i = from; i < edge_count; i += step) {
        uint64_t tmp = edges[i];
        edges_first[i] = FIRST(tmp); edges_second[i] = SECOND(tmp);
    }
}

__global__ void node_index_construct_kernel(const uint32_t *edges_first, uint64_t edge_count, uint32_t *node_index, uint32_t node_count, uint32_t start_node) {
    auto from = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;
    for (uint64_t i = from; i <= edge_count; i += step) {
        int64_t prev = i > 0 ? (int64_t) (edges_first[i - 1] - start_node) : -1;
        int64_t next = i < edge_count ? (int64_t) (edges_first[i] - start_node) : node_count;
        for (int64_t j = prev + 1; j <= next; ++j) { node_index[j] = i; }
    }
}

struct is_self_loop {
    __host__ __device__
    bool operator()(uint64_t x) const { return FIRST(x) == SECOND(x); }
};

void split(uint64_t *edges, uint64_t edge_count, uint64_t part_num, uint32_t **node_index, const uint32_t *node_split, uint64_t *adj_count) {
    uint64_t begin = 0; uint64_t end; uint64_t adj_count_all = 0;
    auto *adj = reinterpret_cast<uint32_t *>(edges);
    uint64_t *dev_edges; uint32_t *dev_edges_first; uint32_t *dev_edges_second; uint32_t *dev_node_index;

    for (uint64_t i = 0; i < part_num; i++) {
        uint32_t stop_node = node_split[i + 1];
        if (i == part_num - 1) { end = edge_count; } else {
            end = swap_if(edges + begin, edges + edge_count, [&](const uint64_t edge) { return FIRST(edge) < stop_node; }) - edges;
        }
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
        CUDA_TRY(cudaDeviceSynchronize());
        CUDA_TRY(cudaMemcpy(adj + adj_count_all, dev_edges_second, adj_count[i] * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t node_count = node_split[i + 1] - node_split[i] + 1;
        uint32_t start_node = node_split[i];
        CUDA_TRY(cudaMalloc((void **) &dev_node_index, (node_count + 1) * sizeof(uint32_t)));
        node_index_construct_kernel<<<numBlocks, BLOCKSIZE>>>(dev_edges_first, adj_count[i], dev_node_index, node_count, start_node);
        CUDA_TRY(cudaDeviceSynchronize());
        node_index[i] = (uint32_t *) malloc((node_count + 1) * sizeof(uint32_t));
        CUDA_TRY(cudaMemcpy(node_index[i], dev_node_index, (node_count + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        adj_count_all += adj_count[i];

        CUDA_TRY(cudaFree(dev_edges)); CUDA_TRY(cudaFree(dev_edges_first));
        CUDA_TRY(cudaFree(dev_edges_second)); CUDA_TRY(cudaFree(dev_node_index));
        begin = end;
    }
    std::vector<uint64_t> adj_count_vec(adj_count, adj_count + part_num);
    adj_count[0] = 0;
    for (uint64_t i = 1; i <= part_num; i++) { adj_count[i] = adj_count[i - 1] + adj_count_vec[i - 1]; }
}

__global__ void node_index_reconstruct_kernel(uint32_t *edges_first, const uint32_t *node_index, uint32_t node_count) {
    auto from = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;
    for (uint64_t i = from; i < node_count; i += step) {
        for (uint64_t j = node_index[i]; j < node_index[i + 1]; ++j) { edges_first[j] = i; }
    }
}

__global__ void warp_binary_kernel(const uint32_t* __restrict__ edge_m, const uint32_t* __restrict__ node_index_m, uint32_t edge_m_count, uint32_t* __restrict__ adj_m, uint32_t start_node_n, const uint32_t* __restrict__ node_index_n, uint32_t node_index_n_count, uint32_t* __restrict__ adj_n, uint64_t *results) {
    uint64_t count = 0;
    __shared__ uint32_t local[BLOCKSIZE];
    uint32_t i = threadIdx.x % 32;
    uint32_t p = threadIdx.x / 32;
    for (uint32_t tid = (threadIdx.x + blockIdx.x * blockDim.x) / 32; tid < edge_m_count; tid += blockDim.x * gridDim.x / 32) {
        uint32_t node_m = edge_m[tid];
        uint32_t node_n = adj_m[tid];
        if (node_n < start_node_n || node_n >= start_node_n + node_index_n_count) continue;

        uint32_t degree_m = node_index_m[node_m + 1] - node_index_m[node_m];
        uint32_t degree_n = node_index_n[node_n + 1 - start_node_n] - node_index_n[node_n - start_node_n];
        uint32_t* a = adj_m + node_index_m[node_m];
        uint32_t* b = adj_n + node_index_n[node_n - start_node_n];
        if(degree_m < degree_n){
            uint32_t temp = degree_m; degree_m = degree_n; degree_n = temp;
            uint32_t *aa = a; a = b; b = aa;
        }
        local[p * 32 + i] = a[i * degree_m / 32];
        __syncthreads();

        uint32_t j = i;
        while(j < degree_n){
            uint32_t X = b[j]; uint32_t Y;
            int32_t bot = 0; int32_t top = 32; int32_t r;
            while(top > bot + 1){
                r = (top + bot) / 2; Y = local[p * 32 + r];
                if(X == Y){ count++; bot = top + 32; }
                if(X < Y){ top = r; }
                if(X > Y){ bot = r; }
            }
            bot = bot * degree_m / 32; top = top * degree_m / 32 - 1;
            while(top >= bot){
                r = (top + bot) / 2; Y = a[r];
                if(X == Y){ count++; }
                if(X <= Y){ top = r - 1; }
                if(X >= Y){ bot = r + 1; }
            }
            j += 32;
        }
        uint64_t warp_sum = count;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) { warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset); }
        if (i == 0) { results[tid] = warp_sum; }
        count = 0;
        __syncthreads();
    }
}

__global__ void init_exact_weights_kernel(
    const uint32_t* edge_m, const uint32_t* adj_m, const uint32_t* undir_node_index, const uint32_t* undir_adj,
    double* exact_tri_counts, uint32_t edge_count
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;

    uint32_t u = edge_m[i]; uint32_t v = adj_m[i];
    uint32_t ptr_u = undir_node_index[u]; uint32_t end_u = undir_node_index[u+1];
    uint32_t ptr_v = undir_node_index[v]; uint32_t end_v = undir_node_index[v+1];

    double deg_u = (double)(end_u - ptr_u);
    double deg_v = (double)(end_v - ptr_v);
    double weight_sum = 0.0;
    while (ptr_u < end_u && ptr_v < end_v) {
        uint32_t w_u = undir_adj[ptr_u]; uint32_t w_v = undir_adj[ptr_v];
        if (w_u == w_v) {
            double deg_w = (double)(undir_node_index[w_u+1] - undir_node_index[w_u]);
            weight_sum += 1.0 / (deg_u * deg_v * deg_w);
            ptr_u++; ptr_v++;
        } 
        else if (w_u < w_v) ptr_u++;
        else ptr_v++;
    }
    exact_tri_counts[i] = weight_sum;
}

__global__ void checker_kernel(
    const uint32_t* edge_m, const uint32_t* adj_m, const uint32_t* node_index, const double* tri_counts,
    bool* is_active, bool* just_deleted, double eps, uint32_t edge_count, bool* d_changed
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    if (!is_active[i]) return;

    uint32_t u = edge_m[i]; uint32_t v = adj_m[i];
    double deg_u = (double)(node_index[u+1] - node_index[u]);
    double deg_v = (double)(node_index[v+1] - node_index[v]);
    double threshold = eps * (1.0 / (deg_u * deg_v));

    if (tri_counts[i] < threshold - 1e-9) {
        is_active[i] = false; just_deleted[i] = true; *d_changed = true;
    }
}

__global__ void updater_kernel(
    const uint32_t* edge_m, const uint32_t* adj_m, const uint32_t* undir_node_index, const uint32_t* undir_adj,
    const uint32_t* undir_edge_id, double* tri_counts, bool* is_active, bool* just_deleted, uint32_t edge_count
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    if (!just_deleted[i]) return;

    uint32_t u = edge_m[i]; uint32_t v = adj_m[i];
    uint32_t ptr_u = undir_node_index[u]; uint32_t end_u = undir_node_index[u+1];
    uint32_t ptr_v = undir_node_index[v]; uint32_t end_v = undir_node_index[v+1];

    double deg_u = (double)(end_u - ptr_u);
    double deg_v = (double)(end_v - ptr_v);

    while (ptr_u < end_u && ptr_v < end_v) {
        uint32_t w_u = undir_adj[ptr_u]; uint32_t w_v = undir_adj[ptr_v];
        if (w_u == w_v) {
            uint32_t id_uw = undir_edge_id[ptr_u]; uint32_t id_vw = undir_edge_id[ptr_v]; 
            double deg_w = (double)(undir_node_index[w_u+1] - undir_node_index[w_u]);
            double decrement_weight = 1.0 / (deg_u * deg_v * deg_w);
            bool a_act = is_active[id_uw]; bool a_dead = just_deleted[id_uw];
            bool b_act = is_active[id_vw]; bool b_dead = just_deleted[id_vw];

            if (a_act) { if (b_act || (b_dead && i < id_vw)) { atomicAdd(&tri_counts[id_uw], -decrement_weight); } }
            if (b_act) { if (a_act || (a_dead && i < id_uw)) { atomicAdd(&tri_counts[id_vw], -decrement_weight); } }
            ptr_u++; ptr_v++;
        } 
        else if (w_u < w_v) ptr_u++;
        else ptr_v++;
    }
}

// Forward Declaration
void execute_gpu_peeling(const uint32_t* dev_edge_m, const uint32_t* dev_adj_m, const uint32_t* dev_node_index, uint64_t* dev_tri_counts, uint32_t edge_count, uint32_t node_count, double eps);
void extract_cluster_gunrock(uint32_t seed_node, const uint32_t* dev_undir_node_index, const uint32_t* dev_undir_adj, const bool* dev_is_active, uint32_t node_count, uint32_t edge_count, double eps);

uint64_t tricount_gpu(uint64_t part_num, uint32_t *adj, const uint64_t *adj_count, const uint32_t *node_split, uint32_t **node_index) {
    uint64_t all_sum = 0;
    uint32_t *node_index_m_dev, *adj_m_dev, *edge_m_dev, *node_index_n_dev, *adj_n_dev, *edge_n_dev;
    uint64_t *dev_results;
    for (uint64_t m = 0; m < part_num; m++) {
        uint32_t *node_index_m = node_index[m];
        uint32_t node_index_m_count = node_split[m + 1] - node_split[m];
        if (node_index_m_count == 0) continue;
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
            uint32_t start_node_n = node_split[n]; uint32_t *node_index_n = node_index[n];
            uint32_t node_index_n_count = node_split[n + 1] - node_split[n];
            uint64_t start_adj_n = adj_count[n]; uint32_t *adj_n = adj + start_adj_n;
            uint32_t adj_count_n = adj_count[n + 1] - adj_count[n];
            CUDA_TRY(cudaMalloc((void **) &node_index_n_dev, (node_index_n_count + 1) * sizeof(uint32_t)));
            CUDA_TRY(cudaMemcpy(node_index_n_dev, node_index_n, (node_index_n_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_TRY(cudaMalloc((void **) &adj_n_dev, adj_count_n * sizeof(uint32_t)));
            CUDA_TRY(cudaMemcpy(adj_n_dev, adj_n, adj_count_n * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_TRY(cudaMalloc((void **) &dev_results, adj_count_m * sizeof(uint64_t)));

            warp_binary_kernel<<<numBlocks, BLOCKSIZE>>>(edge_m_dev, node_index_m_dev, adj_count_m, adj_m_dev, start_node_n, node_index_n_dev, node_index_n_count, adj_n_dev, dev_results);
            CUDA_TRY(cudaDeviceSynchronize());

            thrust::device_ptr<uint64_t> ptr(dev_results);
            uint64_t sum = thrust::reduce(ptr, ptr + adj_count_m); 
            all_sum += sum;

            if (m == 0 && n == 0) { 
                execute_gpu_peeling(edge_m_dev, adj_m_dev, node_index_m_dev, dev_results, adj_count_m, node_split[1], 0.1);
            }
            if (m != n) { CUDA_TRY(cudaMalloc((void **) &edge_n_dev, adj_count_n * sizeof(uint32_t))); }
            CUDA_TRY(cudaFree(node_index_n_dev)); CUDA_TRY(cudaFree(adj_n_dev)); CUDA_TRY(cudaFree(dev_results));
        }
        CUDA_TRY(cudaFree(node_index_m_dev)); CUDA_TRY(cudaFree(adj_m_dev)); CUDA_TRY(cudaFree(edge_m_dev));
    }
    return all_sum;
}

__global__ void verify_integer_triangles_kernel(
    const uint32_t* edge_m, const uint32_t* adj_m, const uint32_t* undir_node_index, const uint32_t* undir_adj,
    const uint32_t* undir_edge_id, const bool* is_active, uint32_t edge_count, unsigned long long* total_integer_triangles
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) return;
    if (!is_active[i]) return; 

    uint32_t u = edge_m[i]; uint32_t v = adj_m[i];
    uint32_t ptr_u = undir_node_index[u]; uint32_t end_u = undir_node_index[u+1];
    uint32_t ptr_v = undir_node_index[v]; uint32_t end_v = undir_node_index[v+1];
    unsigned long long local_count = 0;

    while (ptr_u < end_u && ptr_v < end_v) {
        uint32_t w_u = undir_adj[ptr_u]; uint32_t w_v = undir_adj[ptr_v];
        if (w_u == w_v) {
            uint32_t id_uw = undir_edge_id[ptr_u]; uint32_t id_vw = undir_edge_id[ptr_v];
            if (is_active[id_uw] && is_active[id_vw]) { local_count++; }
            ptr_u++; ptr_v++;
        } 
        else if (w_u < w_v) ptr_u++;
        else ptr_v++;
    }
    if (local_count > 0) { atomicAdd(total_integer_triangles, local_count); }
}

// =========================================================================
// >>> PHASE 2: PURE CUDA CLUSTER EXTRACTION & GREEDY SWEEP <<<
// =========================================================================

__global__ void get_active_frontier_kernel(
    uint32_t seed_node, uint32_t seed_deg, double eps,
    const uint32_t* node_index, const uint32_t* adj, 
    const uint32_t* edge_id, const bool* is_active, 
    uint32_t* frontier, uint32_t* count
) {
    uint32_t start = node_index[seed_node];
    uint32_t end = node_index[seed_node+1];
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (start + tid < end) {
        uint32_t e_idx = start + tid;
        uint32_t v = adj[e_idx];
        uint32_t deg_v = node_index[v+1] - node_index[v];
        
        // THE FIX: Serial Code Degree Filter
        if (is_active[edge_id[e_idx]] && (double)deg_v <= (double)seed_deg / eps) {
            uint32_t pos = atomicAdd(count, 1);
            frontier[pos] = v;
        }
    }
}

__global__ void get_2hop_candidates_kernel(
    uint32_t seed_deg, double eps,
    const uint32_t* node_index, const uint32_t* adj, const uint32_t* edge_id, const bool* is_active,
    const uint32_t* frontier, uint32_t frontier_size,
    uint32_t* candidates, uint32_t* count, uint32_t max_safe_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    uint32_t u = frontier[tid];
    uint32_t start = node_index[u];
    uint32_t end = node_index[u+1];

    for (uint32_t i = start; i < end; i++) {
        if (is_active[edge_id[i]]) {
            uint32_t v = adj[i];
            uint32_t deg_v = node_index[v+1] - node_index[v];
            
            // THE FIX: Serial Code Degree Filter
            if ((double)deg_v <= (double)seed_deg / eps) {
                uint32_t pos = atomicAdd(count, 1);
                if (pos < max_safe_size) candidates[pos] = v;
            }
        }
    }
}

__global__ void compute_tu_and_degree_kernel(
    const uint32_t* candidates, uint32_t num_candidates,
    const uint32_t* frontier, uint32_t frontier_size,
    const uint32_t* node_index, const uint32_t* adj, const uint32_t* edge_id, const bool* is_active,
    uint32_t* tu_array, uint32_t* degree_array
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;

    uint32_t u = candidates[tid];
    uint32_t start = node_index[u];
    uint32_t end = node_index[u+1];
    
    uint32_t active_deg = 0;
    uint32_t tu = 0;
    for(uint32_t i = start; i < end; i++) {
        if (is_active[edge_id[i]]) { 
            active_deg++;
            uint32_t v = adj[i];
            int L = 0, R = frontier_size - 1;
            while(L <= R) {
                int M = L + (R - L) / 2;
                if (frontier[M] == v) { tu++; break; }
                if (frontier[M] < v) L = M + 1;
                else R = M - 1;
            }
        }
    }
    degree_array[tid] = active_deg;
    tu_array[tid] = tu;
}

__global__ void mask_cluster_kernel(
    const uint32_t* cluster_nodes, uint32_t cluster_size,
    const uint32_t* undir_node_index, const uint32_t* undir_edge_id,
    bool* is_active, bool* just_deleted
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cluster_size) return;

    uint32_t u = cluster_nodes[tid];
    uint32_t start = undir_node_index[u];
    uint32_t end = undir_node_index[u+1];

    for(uint32_t i = start; i < end; i++) {
        uint32_t edge = undir_edge_id[i];
        if (is_active[edge]) {
            is_active[edge] = false;
            just_deleted[edge] = true;
        }
    }
}

struct ZipComp {
    __host__ __device__
    bool operator()(const thrust::tuple<uint32_t, uint32_t, uint32_t>& a, 
                    const thrust::tuple<uint32_t, uint32_t, uint32_t>& b) const {
        if (thrust::get<1>(a) == thrust::get<1>(b)) return thrust::get<2>(a) < thrust::get<2>(b); 
        return thrust::get<1>(a) > thrust::get<1>(b); 
    }
};

struct RatioFunctor {
    __host__ __device__
    double operator()(const thrust::tuple<uint32_t, uint32_t>& a) const {
        double num = thrust::get<0>(a);
        double den = thrust::get<1>(a);
        return num / (den + 1e-9);
    }
};

struct MemoryPool {
    uint32_t* dev_frontier;
    uint32_t* pinned_frontier_count; // CPU visible
    uint32_t* dev_frontier_count;    // GPU mapped
    
    uint32_t* dev_candidates;
    uint32_t* pinned_candidate_count; // CPU visible
    uint32_t* dev_candidate_count;    // GPU mapped
    
    uint32_t* dev_tu;
    uint32_t* dev_degree;
    double* dev_ratios;

    bool* pinned_clustered;      // CPU visible tracker
    bool* dev_mapped_clustered;  // GPU mapped tracker
};

// =========================================================================
// >>> PHASE 2: ZERO-COPY HYBRID CLUSTER EXTRACTION <<<
// =========================================================================

uint32_t extract_cluster_gunrock(
    uint32_t seed_node, uint32_t seed_deg, const uint32_t* dev_undir_node_index, const uint32_t* dev_undir_adj,
    const uint32_t* dev_undir_edge_id, bool* dev_is_active, bool* dev_just_deleted,
    uint32_t node_count, double eps, Profiler& prof, MemoryPool& pool
) {
    uint32_t max_possible = seed_deg;
    if (max_possible == 0) {
        pool.pinned_clustered[seed_node] = true;
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    // ZERO LATENCY CPU WRITE
    *pool.pinned_frontier_count = 0; 
    auto t_end = std::chrono::high_resolution_clock::now();
    prof.cpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();

    t_start = std::chrono::high_resolution_clock::now();
    int threads = 256;
    int blocks = (max_possible + threads - 1) / threads;
    if (blocks > 0) {
        get_active_frontier_kernel<<<blocks, threads>>>(seed_node, seed_deg, eps, dev_undir_node_index, dev_undir_adj, dev_undir_edge_id, dev_is_active, pool.dev_frontier, pool.dev_frontier_count);
        CUDA_TRY(cudaDeviceSynchronize());
    }
    t_end = std::chrono::high_resolution_clock::now();
    prof.gpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();

    // ZERO LATENCY CPU READ: No cudaMemcpy!
    uint32_t num_neighbors = *pool.pinned_frontier_count;

    if (num_neighbors == 0) {
        pool.pinned_clustered[seed_node] = true;
        return 1;
    }

    t_start = std::chrono::high_resolution_clock::now();
    thrust::device_ptr<uint32_t> t_front(pool.dev_frontier);
    thrust::sort(t_front, t_front + num_neighbors);
    t_end = std::chrono::high_resolution_clock::now();
    prof.gpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();

    t_start = std::chrono::high_resolution_clock::now();
    // ZERO LATENCY CPU WRITE
    *pool.pinned_candidate_count = 0;
    t_end = std::chrono::high_resolution_clock::now();
    prof.cpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();

    t_start = std::chrono::high_resolution_clock::now();
    uint32_t max_candidates = 50000000; 
    blocks = (num_neighbors + threads - 1) / threads;
    get_2hop_candidates_kernel<<<blocks, threads>>>(
        seed_deg, eps, dev_undir_node_index, dev_undir_adj, dev_undir_edge_id, dev_is_active, 
        thrust::raw_pointer_cast(t_front), num_neighbors, pool.dev_candidates, pool.dev_candidate_count, max_candidates
    );
    CUDA_TRY(cudaDeviceSynchronize());
    t_end = std::chrono::high_resolution_clock::now();
    prof.gpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();

    // ZERO LATENCY CPU READ: No cudaMemcpy!
    uint32_t total_raw_candidates = *pool.pinned_candidate_count;

    t_start = std::chrono::high_resolution_clock::now();
    thrust::device_ptr<uint32_t> cand_ptr(pool.dev_candidates);
    thrust::sort(cand_ptr, cand_ptr + total_raw_candidates);
    auto new_end = thrust::unique(cand_ptr, cand_ptr + total_raw_candidates);
    uint32_t unique_candidate_count = new_end - cand_ptr;
    t_end = std::chrono::high_resolution_clock::now();
    prof.gpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();

    int max_ind = -1;

    if (unique_candidate_count > 0) {
        t_start = std::chrono::high_resolution_clock::now();
        int cand_blocks = (unique_candidate_count + threads - 1) / threads;
        compute_tu_and_degree_kernel<<<cand_blocks, threads>>>(
            thrust::raw_pointer_cast(cand_ptr), unique_candidate_count,
            thrust::raw_pointer_cast(t_front), num_neighbors,
            dev_undir_node_index, dev_undir_adj, dev_undir_edge_id, dev_is_active, pool.dev_tu, pool.dev_degree
        );
        CUDA_TRY(cudaDeviceSynchronize());
        
        auto tu_dptr = thrust::device_pointer_cast(pool.dev_tu);
        auto deg_dptr = thrust::device_pointer_cast(pool.dev_degree);

        auto zip_start = thrust::make_zip_iterator(thrust::make_tuple(cand_ptr, tu_dptr, deg_dptr));
        auto zip_end = zip_start + unique_candidate_count;
        thrust::sort(zip_start, zip_end, ZipComp());

        thrust::inclusive_scan(tu_dptr, tu_dptr + unique_candidate_count, tu_dptr);
        thrust::inclusive_scan(deg_dptr, deg_dptr + unique_candidate_count, deg_dptr);

        auto ratios_dptr = thrust::device_pointer_cast(pool.dev_ratios);
        auto zip_scans = thrust::make_zip_iterator(thrust::make_tuple(tu_dptr, deg_dptr));
        thrust::transform(zip_scans, zip_scans + unique_candidate_count, ratios_dptr, RatioFunctor());

        auto max_iter = thrust::max_element(ratios_dptr, ratios_dptr + unique_candidate_count);
        max_ind = thrust::distance(ratios_dptr, max_iter);

        if (max_ind >= 0) {
            int b_cand = (max_ind + 1 + threads - 1) / threads;
            // The GPU directly writes the tracker updates via dev_mapped_clustered
            mask_cluster_kernel<<<b_cand, threads>>>(thrust::raw_pointer_cast(cand_ptr), max_ind + 1, dev_undir_node_index, dev_undir_edge_id, dev_is_active, dev_just_deleted);
            CUDA_TRY(cudaDeviceSynchronize());
        }
        t_end = std::chrono::high_resolution_clock::now();
        prof.gpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();
    }

    t_start = std::chrono::high_resolution_clock::now();
    pool.pinned_clustered[seed_node] = true;

    int blocks_front = (num_neighbors + threads - 1) / threads;
    if (blocks_front > 0) mask_cluster_kernel<<<blocks_front, threads>>>(thrust::raw_pointer_cast(t_front), num_neighbors, dev_undir_node_index, dev_undir_edge_id, dev_is_active, dev_just_deleted);
    CUDA_TRY(cudaDeviceSynchronize());
    t_end = std::chrono::high_resolution_clock::now();
    prof.gpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();

    return 1 + num_neighbors + (max_ind >= 0 ? max_ind + 1 : 0);
}

void execute_gpu_peeling(
    const uint32_t* dev_edge_m, const uint32_t* dev_adj_m, const uint32_t* dev_node_index,
    uint64_t* dev_tri_counts, uint32_t edge_count, uint32_t node_count, double eps
) {
    Profiler prof;

    int threads = 512; int blocks = (edge_count + threads - 1) / threads;
    uint32_t *host_undir_node_index, *host_undir_adj, *host_undir_edge_id;
    uint32_t* host_dag_edge_m = (uint32_t*)malloc(edge_count * sizeof(uint32_t));
    uint32_t* host_dag_adj_m = (uint32_t*)malloc(edge_count * sizeof(uint32_t));
    CUDA_TRY(cudaMemcpy(host_dag_edge_m, dev_edge_m, edge_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(host_dag_adj_m, dev_adj_m, edge_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    build_undirected_csr_cpu(host_dag_edge_m, host_dag_adj_m, edge_count, node_count, &host_undir_node_index, &host_undir_adj, &host_undir_edge_id);

    uint32_t *dev_undir_node_index, *dev_undir_adj, *dev_undir_edge_id;
    CUDA_TRY(cudaMalloc((void**)&dev_undir_node_index, (node_count + 1) * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc((void**)&dev_undir_adj, 2 * edge_count * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc((void**)&dev_undir_edge_id, 2 * edge_count * sizeof(uint32_t)));

    CUDA_TRY(cudaMemcpy(dev_undir_node_index, host_undir_node_index, (node_count + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(dev_undir_adj, host_undir_adj, 2 * edge_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(dev_undir_edge_id, host_undir_edge_id, 2 * edge_count * sizeof(uint32_t), cudaMemcpyHostToDevice));

    double* dev_exact_tri_counts;
    CUDA_TRY(cudaMalloc((void**)&dev_exact_tri_counts, edge_count * sizeof(double)));

    init_exact_weights_kernel<<<blocks, threads>>>(dev_edge_m, dev_adj_m, dev_undir_node_index, dev_undir_adj, dev_exact_tri_counts, edge_count);
    CUDA_TRY(cudaDeviceSynchronize());

    bool *dev_is_active, *dev_just_deleted, *dev_changed;
    CUDA_TRY(cudaMalloc((void**)&dev_is_active, edge_count * sizeof(bool)));
    CUDA_TRY(cudaMalloc((void**)&dev_just_deleted, edge_count * sizeof(bool)));
    CUDA_TRY(cudaMalloc((void**)&dev_changed, sizeof(bool)));
    CUDA_TRY(cudaMemset(dev_is_active, 1, edge_count * sizeof(bool)));
    CUDA_TRY(cudaMemset(dev_just_deleted, 0, edge_count * sizeof(bool)));

    bool host_changed = true;
    while (host_changed) {
        host_changed = false;
        CUDA_TRY(cudaMemcpy(dev_changed, &host_changed, sizeof(bool), cudaMemcpyHostToDevice));
        checker_kernel<<<blocks, threads>>>(dev_edge_m, dev_adj_m, dev_undir_node_index, dev_exact_tri_counts, dev_is_active, dev_just_deleted, eps, edge_count, dev_changed);
        CUDA_TRY(cudaDeviceSynchronize());
        CUDA_TRY(cudaMemcpy(&host_changed, dev_changed, sizeof(bool), cudaMemcpyDeviceToHost));
        if (host_changed) {
            updater_kernel<<<blocks, threads>>>(dev_edge_m, dev_adj_m, dev_undir_node_index, dev_undir_adj, dev_undir_edge_id, dev_exact_tri_counts, dev_is_active, dev_just_deleted, edge_count);
            CUDA_TRY(cudaDeviceSynchronize());
            CUDA_TRY(cudaMemset(dev_just_deleted, 0, edge_count * sizeof(bool)));
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<uint32_t, uint32_t>> deg_info(node_count);
    for(uint32_t i=0; i<node_count; i++) {
        deg_info[i] = {i, host_undir_node_index[i+1] - host_undir_node_index[i]};
    }
    std::sort(deg_info.begin(), deg_info.end(), [](const auto& a, const auto& b){
        if (a.second == b.second) return a.first < b.first;
        return a.second < b.second;
    });
    auto t_end = std::chrono::high_resolution_clock::now();
    prof.cpu_compute_time += std::chrono::duration<double>(t_end - t_start).count();

    // --- INITIALIZE THE ZERO-COPY MEMORY POOL ---
    MemoryPool pool;
    uint32_t max_candidates = 50000000;
    
    CUDA_TRY(cudaMalloc(&pool.dev_frontier, node_count * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc(&pool.dev_candidates, max_candidates * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc(&pool.dev_tu, max_candidates * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc(&pool.dev_degree, max_candidates * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc(&pool.dev_ratios, max_candidates * sizeof(double)));

    // Allocate Pinned Memory for instant Host-Device transfers
    CUDA_TRY(cudaHostAlloc((void**)&pool.pinned_frontier_count, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_TRY(cudaHostGetDevicePointer((void**)&pool.dev_frontier_count, pool.pinned_frontier_count, 0));

    CUDA_TRY(cudaHostAlloc((void**)&pool.pinned_candidate_count, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_TRY(cudaHostGetDevicePointer((void**)&pool.dev_candidate_count, pool.pinned_candidate_count, 0));

    CUDA_TRY(cudaHostAlloc((void**)&pool.pinned_clustered, node_count * sizeof(bool), cudaHostAllocMapped));
    memset(pool.pinned_clustered, 0, node_count * sizeof(bool));
    CUDA_TRY(cudaHostGetDevicePointer((void**)&pool.dev_mapped_clustered, pool.pinned_clustered, 0));

    uint32_t total_nontrivial_clustered_vertices = 0;

    for (uint32_t i = 0; i < node_count; i++) {
        uint32_t seed_node = deg_info[i].first;
        uint32_t seed_deg = deg_info[i].second;
        
        // ZERO LATENCY CPU CHECK
        if (pool.pinned_clustered[seed_node]) continue;

        uint32_t c_size = extract_cluster_gunrock(
            seed_node, seed_deg, dev_undir_node_index, dev_undir_adj, dev_undir_edge_id,
            dev_is_active, dev_just_deleted, node_count, eps, prof, pool
        );
        
        if (c_size > 1) {
            total_nontrivial_clustered_vertices += c_size;
        }
    }

    CUDA_TRY(cudaFree(pool.dev_frontier)); CUDA_TRY(cudaFreeHost(pool.pinned_frontier_count));
    CUDA_TRY(cudaFree(pool.dev_candidates)); CUDA_TRY(cudaFreeHost(pool.pinned_candidate_count));
    CUDA_TRY(cudaFree(pool.dev_tu)); CUDA_TRY(cudaFree(pool.dev_degree));
    CUDA_TRY(cudaFree(pool.dev_ratios)); CUDA_TRY(cudaFreeHost(pool.pinned_clustered));

    printf("\n--------------------------------------------\n");
    printf("Generating details of RTRex-example decomposition\n");
    printf("Non-trivial clustered vertices: %u\n", total_nontrivial_clustered_vertices);
    printf("--------------------------------------------\n");
    
    printf("\n============================================\n");
    printf(">>> ARCHITECTURAL BOTTLENECK PROFILE <<<\n");
    printf("============================================\n");
    printf("1. VRAM Malloc/Free Time: %f seconds\n", prof.vram_alloc_time);
    printf("2. PCIe Transfer Time:    %f seconds\n", prof.pcie_transfer_time);
    printf("3. GPU Compute Time:      %f seconds\n", prof.gpu_compute_time);
    printf("4. CPU Compute Time:      %f seconds\n", prof.cpu_compute_time);
    printf("============================================\n\\n");

    free(host_dag_edge_m); free(host_dag_adj_m);
    CUDA_TRY(cudaFree(dev_is_active)); CUDA_TRY(cudaFree(dev_just_deleted)); CUDA_TRY(cudaFree(dev_changed));
    CUDA_TRY(cudaFree(dev_exact_tri_counts)); CUDA_TRY(cudaFree(dev_undir_node_index));
    CUDA_TRY(cudaFree(dev_undir_adj)); CUDA_TRY(cudaFree(dev_undir_edge_id));
}