#pragma once

#include <filesystem>
#include <regex>
#include <string>
#include <vector>

#include "../utils/format.hpp"
#include "../utils/system.hpp"

namespace deep_ep::jit {

/*
 * JIT include hash 解析器。
 *
 * 问题:
 *   generated source 本身很短，但真正 kernel 逻辑在 <deep_ep/...> include 里。
 *   如果只 hash generated source，修改 impl/header 后可能错误复用旧 cubin。
 *
 * 方案:
 *   递归解析标准格式 include:
 *
 *       #include <deep_ep/...>
 *
 *   对每个 include 文件内容和其子 include 计算 hash，拼到 source 前作为注释，
 *   最终参与 Compiler::build 的 cache key。
 */

class IncludeParser {
    std::unordered_map<std::string, std::optional<std::string>> cache;

    static std::vector<std::string> get_includes(const std::string& code, const std::filesystem::path& file_path = "") {
        std::vector<std::string> includes;
        const std::regex pattern(R"(#\s*include\s*[<"][^>"]+[>"])");
        std::sregex_iterator iter(code.begin(), code.end(), pattern);
        const std::sregex_iterator end;

        // 只接受规范的 #include <deep_ep/...>，避免 JIT cache 漏掉相对路径依赖。
        for (; iter != end; ++ iter) {
            const auto include_str = iter->str();
            const int len = include_str.length();
            if (include_str.substr(0, 10) == "#include <" and include_str[len - 1] == '>' and include_str[10] != ' ' and include_str[len - 2] != ' ') {
                std::string filename = include_str.substr(10, len - 11);
                if (filename.substr(0, 7) == "deep_ep")  // We only parse `<deep_ep/*>`
                    includes.push_back(filename);
            } else {
                std::string error_info = fmt::format("Non-standard include: {}", include_str);
                if (file_path != "")
                    error_info += fmt::format(" ({})", file_path.string());
                EP_HOST_UNREACHABLE(error_info);
            }
        }
        return includes;
    }

public:
    static std::filesystem::path library_include_path;

    static void prepare_init(const std::string& library_root_path) {
        library_include_path = std::filesystem::path(library_root_path) / "include";
    }

    std::string get_hash_value(const std::string& code, const bool& exclude_code = true) {
        std::stringstream ss;
        for (const auto& i: get_includes(code))
            ss << get_hash_value_by_path(library_include_path / i) << "$";
        if (not exclude_code)
            ss << "#" << get_hex_digest(code);
        return get_hex_digest(ss.str());
    }

    std::string get_hash_value_by_path(const std::filesystem::path& path) {
        // cache[path] = nullopt 表示当前递归栈正在解析该文件，用来检测 circular include。
        // ReSharper disable once CppUseAssociativeContains
        if (cache.count(path) > 0) {
            const auto opt = cache[path];
            if (not opt.has_value())
                EP_HOST_UNREACHABLE(fmt::format("Circular include may occur: {}", path.string()));
            return opt.value();
        }

        // Read file and calculate hash recursively
        std::ifstream in(path);
        if (not in.is_open())
            EP_HOST_UNREACHABLE(fmt::format("Failed to open: {}", path.string()));
        std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        cache[path] = std::nullopt;
        return (cache[path] = get_hash_value(code, false)).value();
    }
};

EP_DECLARE_STATIC_VAR_IN_CLASS(IncludeParser, library_include_path);

static auto include_parser = std::make_shared<IncludeParser>();

}  // namespace deep_ep::jit
