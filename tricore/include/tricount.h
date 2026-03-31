#pragma once

#include "util.h"

using namespace std;

uint64_t init_gpu();

void cal_degree(const uint64_t *edges, uint64_t edge_count, uint32_t *deg, uint32_t node_count);

void redirect_edges(uint64_t *edges, uint64_t edge_count, const uint32_t *deg, uint32_t node_count);

uint32_t cal_part_num(const uint64_t *edges, uint64_t edge_count, uint32_t node_count);

void split(uint64_t *edges, uint64_t edge_count, uint64_t part_num, uint32_t **node_index, const uint32_t *node_split, uint64_t *adj_count);

uint64_t tricount_gpu(uint64_t part_num, uint32_t *adj, const uint64_t *adj_count, const uint32_t *node_split, uint32_t **node_index);

// void execute_gpu_peeling(
//     const uint64_t* dev_edges,
//     const uint32_t* dev_node_index,
//     const uint32_t* dev_adj,
//     const uint32_t* dev_degrees,
//     uint64_t* dev_tri_counts,
//     uint64_t edge_count,
//     float eps
// );
void execute_gpu_peeling(
    const uint32_t* dev_edge_m,
    const uint32_t* dev_adj_m,
    const uint32_t* dev_node_index,
    uint64_t* dev_tri_counts,
    uint32_t edge_count,
    uint32_t node_count,
    double eps
);