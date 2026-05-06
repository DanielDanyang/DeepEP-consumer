#pragma once

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>
#include <optional>
#include <regex>
#include <string>

#include <deep_ep/common/exception.cuh>

#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"

namespace deep_ep::jit {

/*
 * JIT compiler。
 *
 * v2 launch wrapper 生成的源码很小，通常只有:
 *
 *     #include <deep_ep/impls/foo.cuh>
 *     static void __instantiate_kernel() {
 *         auto ptr = reinterpret_cast<void*>(&foo_impl<template args...>);
 *     }
 *
 * Compiler 负责把这段源码编译成 cubin，并用稳定 cache key 复用结果。
 *
 * cache key:
 *
 *     kernel name + compiler signature + flags + generated source
 *
 * 磁盘流程:
 *
 *     build(name, code)
 *          |
 *          +-- process/disk cache hit -> KernelRuntime
 *          |
 *          +-- miss:
 *                tmp dir / kernel.cu
 *                nvcc -> kernel.cubin
 *                optional ptx/sass dump
 *                fsync tmp dir
 *                atomic rename tmp -> cache/kernel.<name>.<hash>
 *                load KernelRuntime
 *
 * 使用目录级 atomic rename 是为了多 rank 同时编译时不读到半写入文件。
 */

class Compiler {
public:
    static std::filesystem::path library_root_path;
    static std::filesystem::path library_include_path;
    static std::filesystem::path cuda_home;
    static std::filesystem::path nccl_root;
    static std::filesystem::path cuobjdump_path;

    static void prepare_init(const std::string& library_root_path,
                             const std::string& cuda_home_path_by_python,
                             const std::string& nccl_root_path_by_python) {
        // 如果 JIT kernel 新增第三方 include，需要把对应内容纳入 hash，否则 cache 可能误命中。
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::cuda_home = cuda_home_path_by_python;
        Compiler::nccl_root = nccl_root_path_by_python;
        Compiler::cuobjdump_path = Compiler::cuda_home / "bin" / "cuobjdump";
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        EP_HOST_ASSERT(not library_root_path.empty());
        EP_HOST_ASSERT(not library_include_path.empty());
        EP_HOST_ASSERT(not cuda_home.empty());
        EP_HOST_ASSERT(not nccl_root.empty());
        EP_HOST_ASSERT(not cuobjdump_path.empty());

        // 默认 cache 在 ~/.deep_ep，可用 EP_JIT_CACHE_DIR 覆盖。
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_ep";
        if (const auto env_cache_dir_path = get_env<std::string>("EP_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // 基础 flags 会进入 signature，因此环境变量变化会导致新的 cache key。
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940,3012 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("EP_JIT_CPP_STANDARD", 20));
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PTXAS_VERBOSE", 0))
            flags += " --ptxas-options=--verbose";
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
        if (get_env("EP_GIN_GDAKI_DEBUG", 0))
            flags += " -DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1";
        flags += fmt::format(" -I {}/include", nccl_root.c_str());

        // Some special flags for EP
        // TODO: make it more general, e.g. `EP_JIT_EXTRA_FLAGS`
        if (int num_topk_idx_bits = get_env("EP_NUM_TOPK_IDX_BITS", 0); num_topk_idx_bits != 0)
            flags += fmt::format(" -DEP_NUM_TOPK_IDX_BITS={}", num_topk_idx_bits);
    }

    virtual ~Compiler() = default;

    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    static void fsync_path(const std::filesystem::path& path) {
        const auto fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::fsync(fd);
            ::close(fd);
        }
    }

    // 递归 fsync: 先文件/子目录，再目录本身。分布式文件系统上 close() 不一定足够。
    static void fsync_dir(const std::filesystem::path& dir_path) { // NOLINT(*-no-recursion)
        for (const auto& entry: std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_directory())
                fsync_dir(entry.path());
            else if (entry.is_regular_file())
                fsync_path(entry.path());
        }
        fsync_path(dir_path);
    }

    static void put(const std::filesystem::path& path, const std::string& data) {
        std::ofstream out(path, std::ios::binary);
        EP_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // NOTES: fsync to ensure the data is visible to other processes (e.g., NVCC)
        // on distributed filesystems, where `close()` alone does not guarantee persistence
        fsync_path(path);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // 先查进程内 cache / 磁盘 cache。
        if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // miss 后在 tmp dir 中完整编译，再原子 rename 成最终 cache dir。
        const auto tmp_dir_path = make_tmp_dir() / get_uuid();
        make_dirs(tmp_dir_path);

        // Compile into the temporary directory
        const auto tmp_cubin_path = tmp_dir_path / "kernel.cubin";
        if (get_env<int>("EP_JIT_DUMP_ASM") or get_env<int>("EP_JIT_DUMP_PTX")) {
            const auto tmp_ptx_path = tmp_dir_path / "kernel.ptx";
            compile(code, tmp_dir_path, tmp_cubin_path, tmp_ptx_path);
        } else {
            compile(code, tmp_dir_path, tmp_cubin_path);
        }

        // Disassemble if needed
        if (get_env<int>("EP_JIT_DUMP_ASM") or get_env<int>("EP_JIT_DUMP_SASS")) {
            const auto tmp_sass_path = tmp_dir_path / "kernel.sass";
            disassemble(tmp_cubin_path, tmp_sass_path);
        }

        // rename 前 fsync，保证其他 rank rename 后能看到完整内容。
        fsync_dir(tmp_dir_path);

        // Atomically rename the temporary directory to the final cache path
        // NOTES: if another rank already created dir_path, rename will fail — that's fine
        make_dirs(dir_path.parent_path());
        std::error_code error_code;
        std::filesystem::rename(tmp_dir_path, dir_path, error_code);
        if (error_code) {
            // 其他 rank 先完成了编译，清理自己的 tmp dir 并复用已有 cache。
            safe_remove_all(tmp_dir_path);
        }

        // Put into the runtime cache
        const auto runtime = kernel_runtime_cache->get(dir_path);
        EP_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    static void disassemble(const std::filesystem::path &cubin_path, const std::filesystem::path &sass_path) {
        // Disassemble the CUBIN file to SASS
        const auto command = fmt::format("{} --dump-sass {} > {}", cuobjdump_path.c_str(), cubin_path.c_str(), sass_path.c_str());
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running cuobjdump command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("cuobjdump failed: %s\n", output.c_str());
            EP_HOST_ASSERT(false and "cuobjdump failed");
        }
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::optional<std::filesystem::path> &ptx_path = std::nullopt) const = 0;
};

EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, nccl_root);
EP_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuobjdump_path);

class NVCCCompiler final: public Compiler {
    std::filesystem::path nvcc_path;

    std::pair<int, int> get_nvcc_version() const {
        EP_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto command = std::string(nvcc_path) + " --version";
        const auto [return_code, output] = call_external_command(command);
        EP_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3
        int major, minor;
        std::smatch match;
        EP_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        EP_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        return {major, minor};
    }

public:
    NVCCCompiler() {
        // NVCC 版本和 GPU arch 都影响 cubin，所以进入 compiler signature/flags。
        nvcc_path = cuda_home / "bin" / "nvcc";
        cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        if (const auto env_nvcc_path = get_env<std::string>("EP_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        const auto [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // NVCC >= 12.9 支持 arch family suffix；Blackwell/SM100 需要 100a/100f 区分。
        const auto arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), arch);
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // 写 kernel.cu，失败时保留 tmp 目录便于 EP_JIT_DEBUG 排查。
        const auto code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // 使用独立 compile_dir，避免当前工作目录里的文件遮蔽 C++ 标准库头。
        const auto compile_dir = make_tmp_dir();
        const auto command = fmt::format("cd {} && {} {} -cubin -o {} {}",
            compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            EP_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Compile to PTX if needed
        if (ptx_path.has_value()) {
            const auto ptx_command = fmt::format("cd {} && {} {} -ptx -o {} {}",
                compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), ptx_path->c_str(), flags);
            if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PRINT_COMPILER_COMMAND", 0))
                printf("Running NVCC PTX command: %s\n", ptx_command.c_str());
            const auto [ptx_return_code, ptx_output] = call_external_command(ptx_command);
            if (ptx_return_code != 0) {
                printf("NVCC PTX compilation failed: %s\n", ptx_output.c_str());
                EP_HOST_ASSERT(false and "NVCC PTX compilation failed");
            }
        }

        // 可选检查 ptxas 是否使用 local memory；高性能 kernel 通常不希望 spill。
        if (get_env("EP_JIT_PTXAS_CHECK", 0))
            EP_HOST_ASSERT(not std::regex_search(output, std::regex(R"(Local memory used)")));

        // Print PTXAS log
        if (get_env("EP_JIT_DEBUG", 0) or get_env("EP_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    return std::make_shared<NVCCCompiler>();
});

} // namespace deep_ep::jit
