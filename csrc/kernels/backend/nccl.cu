#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sstream>

#include <nccl.h>
#include <nccl_device/core.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "api.cuh"
#include "../../utils/system.hpp"


namespace deep_ep::nccl {

/*
 * NCCL backend for DeepEP V2。
 *
 * 初始化链路:
 *
 *     Python ProcessGroup / unique id
 *             |
 *             v
 *     host ncclComm_t
 *             |
 *             v
 *     NCCLSymmetricMemoryContext
 *        - create ncclDevComm_t with Gin contexts/QPs
 *        - query physical topology (LSA/NVLink ranks, RDMA ranks)
 *        - choose logical topology (scaleout, scaleup)
 *        - allocate/register nccl symmetric window
 *        - expose mapped_window_ptr to ElasticBuffer
 *
 * topology:
 *
 *     physical:
 *       num_rdma_ranks = world ranks / LSA ranks
 *       num_nvl_ranks  = LSA ranks
 *
 *     logical allow_hybrid_mode=true:
 *       scaleout = RDMA ranks
 *       scaleup  = NVLink/LSA ranks
 *
 *     logical allow_hybrid_mode=false:
 *       scaleout = 1
 *       scaleup  = all ranks
 */

pybind11::bytearray get_local_unique_id() {
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    std::vector<char> result(sizeof(ncclUniqueId));
    std::memcpy(result.data(), &unique_id, sizeof(ncclUniqueId));
    return {result.data(), result.size()};
}

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx) {
    // Copy unique ID
    ncclUniqueId root_unique_id;
    const auto root_unique_id_str = root_unique_id_bytes.cast<std::string>();
    std::memcpy(&root_unique_id, root_unique_id_str.c_str(), sizeof(ncclUniqueId));

    // Init
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("New NCCL host communicator created (%d/%d)\n", rank_idx, num_ranks);
    return reinterpret_cast<int64_t>(comm);
}

void destroy_nccl_comm(const int64_t& nccl_comm) {
    NCCL_CHECK(ncclCommAbort(reinterpret_cast<ncclComm_t>(nccl_comm)));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("NCCL host communicator aborted\n");
}

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm) {
    // physical domain 直接来自 NCCL team world 和 LSA team。
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    const int num_ranks = ncclTeamWorld(comm).nRanks, num_nvl_ranks = ncclTeamLsa(comm).nRanks;
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    return {num_ranks / num_nvl_ranks, num_nvl_ranks};
}

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {
    // hybrid 打开时把 RDMA 维度暴露为 scale-out；关闭时所有 rank 合并到一个 scale-up 域。
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
}

NCCLSymmetricMemoryContext::NCCLSymmetricMemoryContext(const int64_t& nccl_comm,
                                                       const int& num_ranks, const int& rank_idx,
                                                       const size_t& size, const size_t& alignment,
                                                       const bool& allow_hybrid_mode,
                                                       const int& sl_idx, const int& num_allocated_qps):
    rank_idx(rank_idx), num_ranks(num_ranks), num_allocated_qps(num_allocated_qps) {
    if (get_env("EP_BUFFER_DEBUG", 0)) {
        int nccl_version;
        NCCL_CHECK(ncclGetVersion(&nccl_version));
        printf("DeepEP initialized with NCCL version: %d.%d.%d (loaded library)\n",
               nccl_version / 10000, (nccl_version % 10000) / 100, nccl_version % 100);
    }

    // 复用 Python/ProcessGroup 创建的 host communicator。
    comm = reinterpret_cast<ncclComm_t>(nccl_comm);

    // Print number of allocated QPs
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("EP NCCL device communicator has %d allocated QPs\n", num_allocated_qps);

    // 检查当前 NCCL/网络配置是否支持 device-side Gin。
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    EP_HOST_ASSERT(
        (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE and
        "NCCL GIN is unavailable. This is usually due to a network configuration issue, "
        "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network.");

    // 初始化 device communicator。Gin contexts 约等于可分配 QP 数；
    // elastic kernels 会通过 get_qp_mode 把 warp/channel 映射到这些 contexts。
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {
        reqs.ginContextCount = num_allocated_qps;
        reqs.ginExclusiveContexts = true;
        reqs.ginQueueDepth = 1024;
        reqs.ginTrafficClass = sl_idx;
        // Customized RDMA barrier needs extra signals
        reqs.ginSignalCount = num_ranks + 2 * 2;
        reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL: NCCL_GIN_CONNECTION_FULL;
    }
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    // device communicator 创建后才能可靠得到 LSA/NVLink 域。
    num_nvl_ranks = dev_comm.lsaSize, nvl_rank_idx = dev_comm.lsaRank;
    num_rdma_ranks = num_ranks / num_nvl_ranks, rdma_rank_idx = rank_idx / num_nvl_ranks;
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0 and nvl_rank_idx == rank_idx % num_nvl_ranks);
    EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);

    // 逻辑域映射到全局 rank:
    //
    //     rank_idx = scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx
    if (allow_hybrid_mode) {
        num_scaleout_ranks = num_rdma_ranks, num_scaleup_ranks = num_nvl_ranks;
        scaleout_rank_idx = rdma_rank_idx, scaleup_rank_idx = nvl_rank_idx;
    } else {
        num_scaleout_ranks = 1, num_scaleup_ranks = num_ranks;
        scaleout_rank_idx = 0, scaleup_rank_idx = rank_idx;
    }
    is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

    // 创建 NCCL symmetric memory window。
    //
    // raw_window_ptr:    NCCL 分配/注册的原始指针
    // mapped_window_ptr: 当前 rank 在 LSA 地址空间可访问的指针，ElasticBuffer 使用它
    // NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
    // across all ranks, so no explicit barrier is needed after this call.
    NCCL_CHECK(ncclMemAlloc(&raw_window_ptr, size));
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, size, &window, NCCL_WIN_DEFAULT));
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));

    // 预取所有 LSA peer 的 base pointer，host 侧 get_sym_ptr 可快速计算 peer 地址。
    // TODO: check whether this is correct for network with RDMA
    nvl_window_ptrs.resize(num_nvl_ranks);
    for (int i = 0; i < num_nvl_ranks; ++ i)
        NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));

    // TODO: push NCCL team to support aligned allocation
    EP_HOST_ASSERT(size % alignment == 0);
    EP_HOST_ASSERT(reinterpret_cast<uint64_t>(raw_window_ptr) % alignment == 0);
    EP_HOST_ASSERT(reinterpret_cast<uint64_t>(mapped_window_ptr) % alignment == 0);
}

void* NCCLSymmetricMemoryContext::get_sym_ptr(void* ptr, const int& dst_rank_idx) const {
    // host 侧 peer pointer 翻译，仅适用于 LSA/NVLink peers。
    const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(mapped_window_ptr);
    return static_cast<uint8_t*>(nvl_window_ptrs[dst_rank_idx]) + offset;
}

void NCCLSymmetricMemoryContext::finalize() const {
    // 销毁顺序必须先 deregister/free window，再销毁 device communicator。
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    NCCL_CHECK(ncclMemFree(raw_window_ptr));

    // Destroy device communicator
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
}

}  // namespace deep_ep::nccl
