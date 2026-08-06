#include "ConfigLoader.hpp"

#include <Windows.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cwchar>
#include <filesystem>
#include <spdlog/spdlog.h>

using namespace SmoothTerrain;

void ConfigLoader::loadConfig()
{
    // The INI lives next to the plugin DLL; current_path is the game root at load time
    const auto iniPath = std::filesystem::current_path() / "Data" / "SKSE" / "Plugins" / "SmoothTerrain.ini";

    // Read every key from [General]; each falls back to its default independently. Integer
    // settings are clamped in the float domain BEFORE the int cast: a float beyond int range
    // makes the cast undefined (MSVC yields INT_MIN), which would silently invert an absurd
    // "as high as possible" value into "off".
    s_config.subdivisions = static_cast<int>(std::clamp(
        readIniFloat(iniPath, L"iSubdivisions", DEFAULT_SUBDIVISIONS), 0.0F, static_cast<float>(MAX_SUBDIVISIONS)));
    s_config.maxRise = std::clamp(readIniFloat(iniPath, L"fMaxRise", DEFAULT_MAX_RISE), 0.0F, MAX_RISE_CAP);

    // 0 (or anything below) lifts the distance limit (every loaded quad smooths); any other
    // value means "the player's quad plus surroundings", which cannot be smaller than the
    // 3x3 quad square (radius 2 counting the player's own quad). The upper cap is the game's
    // own loaded-grid size, only known at runtime (see TerrainFalloff); the clamp here only
    // keeps the int cast defined.
    constexpr float SMOOTHED_QUADS_PARSE_CAP = 1000.0F;
    const auto rawQuads = static_cast<int>(
        std::clamp(readIniFloat(iniPath, L"iSmoothedQuads", DEFAULT_SMOOTHED_QUADS), 0.0F, SMOOTHED_QUADS_PARSE_CAP));
    s_config.smoothedQuads = rawQuads <= 0 ? 0 : std::max(rawQuads, MIN_SMOOTHED_QUADS);

    // 0 disables the gradient (straight to vanilla past the full-level square)
    s_config.gradientStep = static_cast<int>(
        std::clamp(readIniFloat(iniPath, L"iGradientStep", DEFAULT_GRADIENT_STEP), 0.0F, SMOOTHED_QUADS_PARSE_CAP));

    // Log the effective values so user reports include them
    spdlog::info("Config Loaded: Subdivisions: {}", s_config.subdivisions);
    spdlog::info("Config Loaded: Max Rise: {}", s_config.maxRise);
    spdlog::info("Config Loaded: Smoothed Quads: {}", s_config.smoothedQuads);
    spdlog::info("Config Loaded: Gradient Step: {}", s_config.gradientStep);
}

auto ConfigLoader::getSubdivisions() -> int { return s_config.subdivisions; }

auto ConfigLoader::getMaxRise() -> float { return s_config.maxRise; }

auto ConfigLoader::getSmoothedQuads() -> int { return s_config.smoothedQuads; }

auto ConfigLoader::getGradientStep() -> int { return s_config.gradientStep; }

auto ConfigLoader::readIniFloat(const std::filesystem::path& path,
                                const wchar_t* key,
                                float defVal) -> float
{
    // check if ini file exists
    if (!std::filesystem::exists(path)) {
        return defVal;
    }

    // Read the raw value string from the [General] section using the Windows API
    std::array<wchar_t, INI_BUFFER_SIZE> buffer {};
    const auto pathStr = path.wstring();
    GetPrivateProfileStringW(L"General", key, L"", buffer.data(), static_cast<DWORD>(buffer.size()), pathStr.c_str());

    // Empty result means the key is missing (or blank); use the default
    if (buffer.at(0) == L'\0') {
        return defVal;
    }

    // Parse as float; wcstof leaves end at the buffer start when nothing was consumed. A
    // non-finite value ("nan", "inf", overflow) would poison the clamps in loadConfig, so it
    // counts as unparsable too.
    wchar_t* end = nullptr;
    const float parsed = std::wcstof(buffer.data(), &end);
    if (end == buffer.data() || !std::isfinite(parsed)) {
        return defVal;
    }
    return parsed;
}
