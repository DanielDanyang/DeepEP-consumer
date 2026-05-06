#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_ep::jit {

/*
 * 进程内 KernelRuntime 缓存。
 *
 * 磁盘 cache 目录保存 kernel.cu/kernel.cubin；本类再在当前进程里缓存已加载的
 * KernelRuntime，避免同一个 specialized kernel 被重复 cuModuleLoad。
 *
 *     cache dir path
 *          |
 *          +-- process cache hit -> existing KernelRuntime
 *          |
 *          +-- disk valid -> load cubin -> cache
 *          |
 *          +-- missing/corrupt -> nullptr / assert
 */

class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;

public:
    KernelRuntimeCache() = default;

    void clear() {
        cache.clear();
    }

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // Hit the runtime cache
        if (const auto iterator = cache.find(dir_path); iterator != cache.end())
            return iterator->second;

        if (KernelRuntime::check_validity(dir_path))
            return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);
        return nullptr;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_ep::jit
