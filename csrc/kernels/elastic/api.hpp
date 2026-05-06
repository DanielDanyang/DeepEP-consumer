#pragma once

#include <deep_ep/common/compiled.cuh>

/*
 * Elastic JIT launch wrapper 聚合头。
 *
 * 这里包含的不是最终 device kernel 主体，而是 host 侧 launch wrapper：
 *
 *     csrc/elastic/buffer.hpp
 *              |
 *              v
 *     launch_dispatch / launch_combine / ...
 *              |
 *              +-- generate_impl(args)  生成 include impl + template instantiation 字符串
 *              +-- jit::compiler->build 编译/缓存 cubin
 *              +-- launch_impl(args)    把 runtime pointer/shape 参数传给 kernel
 *              |
 *              v
 *     deep_ep/include/deep_ep/impls/*.cuh
 */

#include "barrier.hpp"
#include "dispatch.hpp"
#include "combine.hpp"
#include "engram.hpp"
#include "pp_send_recv.hpp"
