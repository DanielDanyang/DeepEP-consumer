#pragma once

#include <memory>
#include <vector>
#include <optional>

#include <nccl.h>
#include <nccl_device.h>

/*
 * Backend API declarations。
 *
 * v2 elastic 主路径主要使用 deep_ep::nccl:
 *
 *     host ncclComm_t
 *          |
 *          v
 *     NCCLSymmetricMemoryContext
 *          |
 *          +-- ncclDevComm_t        device-side Gin communicator
 *          +-- ncclWindow_t         symmetric memory window
 *          +-- mapped_window_ptr    local LSA mapped pointer
 *          +-- nvl_window_ptrs      NVLink peer symmetric pointers
 *
 * legacy V1 仍保留 deep_ep::nvshmem API；elastic 不沿用 legacy buffer/kernel。
 *
 * cuda_driver helpers 用于 host 侧批量写/等 signal，例如 AGRS session 完成同步。
 */

// TODO: make a unified API
namespace deep_ep::nvshmem {

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val,
         const int& rank,
         const int& num_ranks,
         const int& team_split_stride);

void* alloc(const size_t& size, const size_t& alignment);

void free(void* ptr);

void barrier(const bool& with_cpu_sync, const std::optional<cudaStream_t>& stream_opt = std::nullopt);

void finalize();

}  // deep_ep::nvshmem

namespace deep_ep::nccl {

pybind11::bytearray get_local_unique_id();

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx);

void destroy_nccl_comm(const int64_t& nccl_comm);

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm);

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode);

// TODO: make it header only?
struct NCCLSymmetricMemoryContext {
private:
    // 原始 NCCL allocation pointer 只用于 deregister/free；外部使用 mapped_window_ptr。
    void* raw_window_ptr;

public:
    // Global
    int rank_idx;
    int num_ranks;

    // Logical
    int num_scaleout_ranks, num_scaleup_ranks;
    int scaleout_rank_idx, scaleup_rank_idx;

    // Physical
    int num_rdma_ranks, num_nvl_ranks;
    int rdma_rank_idx, nvl_rank_idx;
    bool is_scaleup_nvlink;

    // NCCL handles
    ncclComm_t comm;
    ncclDevComm_t dev_comm;
    ncclWindow_t window;
    void* mapped_window_ptr;
    std::vector<void*> nvl_window_ptrs;

    // Configs
    int num_allocated_qps;

    NCCLSymmetricMemoryContext(const int64_t& nccl_comm,
                               const int& num_ranks, const int& rank_idx,
                               const size_t& size, const size_t& alignment,
                               const bool& allow_hybrid_mode,
                               const int& sl_idx, const int& num_allocated_qps);

    // TODO: finish this with `explicit_destroy`
    // ~NCCLSymmetricMemoryContext();

    void* get_sym_ptr(void* ptr, const int& dst_rank_idx) const;

    void finalize() const;
};

}  // deep_ep::nccl

namespace deep_ep::cuda_driver {

void batched_write(CUstream stream, const std::vector<void*>& ptrs, const int& value);

void batched_wait(CUstream stream, const std::vector<void*>& ptrs, const int& value);

void batched_write_and_wait(CUstream stream, const std::vector<void*>& write_ptrs, const std::vector<void*>& wait_ptrs, const int& value);

}  // namespace deep_ep::cuda_driver
