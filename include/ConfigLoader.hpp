#pragma once

#include <cstddef>
#include <filesystem>

namespace SmoothTerrain {

/**
 * @brief Loads and serves the plugin configuration from Data/SKSE/Plugins/SmoothTerrain.ini
 *
 * The configuration is read once at plugin load (loadConfig) into a static ConfigMap; the
 * getters are plain accessors and never touch the disk. Missing files, missing keys, or
 * unparsable values silently fall back to the compiled-in defaults.
 */
class ConfigLoader {
private:
    //
    // DEFAULT CFG VALUES
    //
    constexpr static float DEFAULT_SUBDIVISIONS = 1.0F; /**< Default subdivision level (iSubdivisions) */
    constexpr static float DEFAULT_SMOOTHNESS = 1.0F; /**< Default interpolation smoothness (fSmoothness) */
    constexpr static float DEFAULT_MAX_RISE = 3.0F; /**< Default upward overshoot allowance (fMaxRise) */
    constexpr static float MAX_RISE_CAP = 100000.0F; /**< Past the tallest landscape relief, so effectively no limit */
    constexpr static float DEFAULT_SMOOTHED_GRIDS = 2.0F; /**< Default smoothed-grid radius (iSmoothedGrids) */

public:
    constexpr static int MIN_SMOOTHED_GRIDS = 2; /**< Smallest non-zero smoothed-grid radius: the player's cell
                                                    plus its full neighbor ring (3x3). A lone smoothed cell
                                                    would put the stitched (unsmoothed) border lines right at
                                                    the player's feet. */

private:

public:
    constexpr static int MAX_SUBDIVISIONS = 3; /**< Hard cap: level 4 would overflow BSTriShape's 16-bit vertex count
                                                  (129x129 = 16641 verts at level 3 is the last that fits) */

private:
    /**
     * @brief ConfigMap structure which holds the configuration values for the plugin
     */
    struct ConfigMap {
        int subdivisions {}; /**< How many times each 128-unit land quad is split in half per axis (0-3) */
        float smoothness {}; /**< 0 = flat bilinear interpolation, 1 = full Catmull-Rom smoothing */
        float maxRise {}; /**< World units the curve may rise above the original verts around it */
        int smoothedGrids {}; /**< Diameter of the smoothed grid square around the player (0 = all loaded grids) */
    };

    static inline ConfigMap s_config; /**< Holds the current configuration values for the plugin */

    //
    // Hardcoded Settings
    //
    constexpr static size_t INI_BUFFER_SIZE = 64; /**< Character buffer size for reading a single INI value */

public:
    /**
     * @brief Loads the configuration values from the SmoothTerrain.ini file and stores them in s_config
     */
    static void loadConfig();

    /**
     * @brief Get the subdivision level
     *
     * @return int Number of times each land quad edge is halved (0 = vanilla mesh, max 3)
     */
    static auto getSubdivisions() -> int;

    /**
     * @brief Get the interpolation smoothness
     *
     * @return float 0 keeps interpolated verts on the vanilla (flat-shaded) surface, 1 applies full
     *         Catmull-Rom curvature between the original verts
     */
    static auto getSmoothness() -> float;

    /**
     * @brief Get how far, in world units, the height spline may rise past the original verts
     *
     * A Catmull-Rom curve normally overshoots past its knots wherever the sampled heights turn
     * or step sharply. Overshooting upward is what lets the new terrain rise into space every
     * surrounding original vertex was below - the space a static mesh resting on the vanilla
     * surface occupies. Capping it holds each interpolated height within this many units of the
     * highest of the four original verts around it, so a static clearing the vanilla mesh by
     * more than that cannot be poked through. Downward overshoot is never limited, since a dip
     * cannot reach anything resting above the surface.
     *
     * The cap is absolute rather than relative to the local relief so that it can be compared
     * against the clearance a static actually has, which does not scale with how dramatic the
     * terrain beneath it happens to be.
     *
     * @return float World units of allowed rise; 0 pins the curve to the surrounding verts and
     *         MAX_RISE_CAP leaves the raw Catmull-Rom tangents untouched
     */
    static auto getMaxRise() -> float;

    /**
     * @brief Get the radius, in grids, of the smoothed square around the player
     *
     * Counts the player's own grid: 2 means the cell the player stands in plus its full ring
     * of neighbors (a 3x3 square, diagonals included), 3 a 5x5 square, and so on - a value of
     * n covers a (2n-1) by (2n-1) square. Cells outside the square keep the vanilla land
     * mesh, and the square moves with the player (see TerrainFalloff). The runtime
     * additionally caps the square at the game's loaded-grid size, since a cell that is not
     * loaded has no mesh to smooth.
     *
     * @return int The radius (>= 2, counting the player's grid), or 0 to smooth every loaded
     *         grid at build time with no distance falloff
     */
    static auto getSmoothedGrids() -> int;

private:
    /**
     * @brief Reads a single float value from the [General] section of an INI file
     *
     * @param path Path to the INI file
     * @param key Name of the key to read
     * @param defVal Value to return when the file or key is missing, or the value does not parse
     * @return float The parsed value, or defVal on any failure
     */
    static auto readIniFloat(const std::filesystem::path& path,
                             const wchar_t* key,
                             float defVal) -> float;
};

} // namespace SmoothTerrain
