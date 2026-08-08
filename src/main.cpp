#include "ConfigLoader.hpp"
#include "DecalFix.hpp"

#include "TerrainCollision.hpp"
#include "TerrainFalloff.hpp"
#include "TerrainSubdivision.hpp"

#include "PCH.h"

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

using namespace SmoothTerrain;
using namespace std::literals;

namespace {

constexpr std::size_t TRAMPOLINE_SIZE = 192; /**< Each patched call site needs 14 bytes; at most seven sites today
                                                  (two builder plus three decal plus two collision on AE, one
                                                  builder site fewer on SE) */

/**
 * @brief Sets up the global log file for the plugin using spdlog
 */
void setupLog()
{
    // Resolve the SKSE log directory (Documents/My Games/.../SKSE)
    auto logsFolder = SKSE::log::log_directory();
    if (!logsFolder) {
        SKSE::stl::report_and_fail("SKSE log_directory not provided, logs disabled.");
    }

    // Create a truncating file sink named after the plugin and make it the default logger
    auto logFilePath = *logsFolder / (std::string(PLUGIN_NAME) + ".log");
    auto fileLogger = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath.string(), true);
    auto logger = std::make_shared<spdlog::logger>("log", std::move(fileLogger));

    // Log everything and flush per message so crashes don't lose the tail of the log
    spdlog::set_default_logger(std::move(logger));
    spdlog::set_level(spdlog::level::info);
    spdlog::flush_on(spdlog::level::info);
}

} // namespace

//
// CommonLibSSE-NG / SKSE Exports
//

SKSEPluginInfo(.Version = REL::Version {0,
                                        1,
                                        0,
                                        0},
               .Name = "SmoothTerrain",
               .Author = "hakasapl",
               .StructCompatibility = SKSE::StructCompatibility::Independent,
               .RuntimeCompatibility = SKSE::VersionIndependence::AddressLibrary)

    SKSEPluginLoad(const SKSE::LoadInterface* skse)
{
    SKSE::Init(skse);
    setupLog();

    const auto version = REL::Module::get().version();
    spdlog::info("{} {} loading (runtime {})", PLUGIN_NAME, PLUGIN_VERSION, version.string("."));

    // Read the INI once, then patch the land builder call sites while nothing is loading yet
    ConfigLoader::loadConfig();
    SKSE::AllocTrampoline(TRAMPOLINE_SIZE);
    TerrainSubdivision::install();
    TerrainFalloff::install();
    TerrainCollision::install();
    DecalFix::install();

    // The falloff's cell sink needs the game's event sources, which only exist once the data
    // handler is up; SKSE announces that moment with kDataLoaded
    SKSE::GetMessagingInterface()->RegisterListener([](SKSE::MessagingInterface::Message* message) -> void {
        if (message != nullptr && message->type == SKSE::MessagingInterface::kDataLoaded) {
            TerrainFalloff::onDataLoaded();
        }
    });

    spdlog::info("{} loaded", PLUGIN_NAME);
    return true;
}
