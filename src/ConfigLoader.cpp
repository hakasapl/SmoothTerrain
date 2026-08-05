#include "ConfigLoader.hpp"

#include <Windows.h>

#include <algorithm>
#include <array>
#include <cwchar>
#include <filesystem>
#include <spdlog/spdlog.h>

using namespace SmoothTerrain;

void ConfigLoader::loadConfig()
{
    // The INI lives next to the plugin DLL; current_path is the game root at load time
    const auto iniPath = std::filesystem::current_path() / "Data" / "SKSE" / "Plugins" / "SmoothTerrain.ini";

    // Read every key from [General]; each falls back to its default independently
    s_config.subdivisions = std::clamp(
        static_cast<int>(readIniFloat(iniPath, L"iSubdivisions", DEFAULT_SUBDIVISIONS)), 0, MAX_SUBDIVISIONS);
    s_config.smoothness = std::clamp(readIniFloat(iniPath, L"fSmoothness", DEFAULT_SMOOTHNESS), 0.0F, 1.0F);
    s_config.maxRise = std::clamp(readIniFloat(iniPath, L"fMaxRise", DEFAULT_MAX_RISE), 0.0F, MAX_RISE_CAP);

    // 0 (or anything below) disables the falloff outright; any other value means "the player's
    // cell plus surroundings", which cannot be smaller than the 3x3 square (radius 2 counting
    // the player's own grid). The upper cap is the game's own loaded-grid size, only known at
    // runtime (see TerrainFalloff).
    const auto rawGrids = static_cast<int>(readIniFloat(iniPath, L"iSmoothedGrids", DEFAULT_SMOOTHED_GRIDS));
    s_config.smoothedGrids = rawGrids <= 0 ? 0 : std::max(rawGrids, MIN_SMOOTHED_GRIDS);

    // Log the effective values so user reports include them
    spdlog::info("Config Loaded: Subdivisions: {}", s_config.subdivisions);
    spdlog::info("Config Loaded: Smoothness: {}", s_config.smoothness);
    spdlog::info("Config Loaded: Max Rise: {}", s_config.maxRise);
    spdlog::info("Config Loaded: Smoothed Grids: {}", s_config.smoothedGrids);
}

auto ConfigLoader::getSubdivisions() -> int { return s_config.subdivisions; }

auto ConfigLoader::getSmoothness() -> float { return s_config.smoothness; }

auto ConfigLoader::getMaxRise() -> float { return s_config.maxRise; }

auto ConfigLoader::getSmoothedGrids() -> int { return s_config.smoothedGrids; }

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

    // Parse as float; wcstof leaves end at the buffer start when nothing was consumed
    wchar_t* end = nullptr;
    const float parsed = std::wcstof(buffer.data(), &end);
    if (end == buffer.data()) {
        return defVal;
    }
    return parsed;
}
