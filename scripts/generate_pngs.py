import sys
import cfgrib
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from scipy.interpolate import griddata
from zoneinfo import ZoneInfo
from adjustText import adjust_text
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import warnings
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from multiprocessing import Pool
from matplotlib.tri import Triangulation

warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------
# Eingabe-/Ausgabe
# ------------------------------
data_dir = sys.argv[1]
output_dir = sys.argv[2]
var_type = sys.argv[3].lower()
gridfile = sys.argv[4] if len(sys.argv) > 4 else "data/grid/grid.nc"

if not os.path.exists(gridfile):
    raise FileNotFoundError(f"Grid-Datei nicht gefunden: {gridfile}")
    
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Geo-Daten
# ------------------------------
cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden',
             'Stuttgart', 'Düsseldorf', 'Nürnberg', 'Erfurt', 'Leipzig',
             'Bremen', 'Saarbrücken', 'Hannover'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05, 48.78, 51.23,
            49.45, 50.98, 51.34, 53.08, 49.24, 52.37],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73, 9.18, 6.78,
            11.08, 11.03, 12.37, 8.80, 6.99, 9.73]
})

# ------------------------------
# Farben und Normen
# ------------------------------
ww_colors_base = {
    0: "#696969", 1: "#696969", 2: "#696969", 3: "#696969",
    45: "#FFFF00", 48: "#FFD700",
    56: "#FFA500", 57: "#C06A00",
    51: "#00FF00", 53: "#00C300", 55: "#009700",
    61: "#00FF00", 63: "#00C300", 65: "#009700",
    80: "#00FF00", 81: "#00C300", 82: "#009700",
    66: "#FF6347", 67: "#8B0000",
    71: "#ADD8E6", 73: "#6495ED", 75: "#00008B",
    95: "#FF77FF", 96: "#C71585", 99: "#C71585"
}
ww_categories = {
    "Nebel": [45],
    "Schneeregen": [56, 57],
    "Regen": [61, 63, 65],
    "gefr. Regen": [66, 67],
    "Schnee": [71, 73, 75],
    "Gewitter": [95,96],
}

t2m_bounds = list(range(-36, 50, 2))
t2m_colors = LinearSegmentedColormap.from_list(
    "t2m_smoooth",
    [
        "#F675F4", "#F428E9", "#B117B5", "#950CA2", "#640180",
        "#3E007F", "#00337E", "#005295", "#1292FF", "#49ACFF",
        "#8FCDFF", "#B4DBFF", "#B9ECDD", "#88D4AD", "#07A125",
        "#3FC107", "#9DE004", "#E7F700", "#F3CD0A", "#EE5505",
        "#C81904", "#AF0E14", "#620001", "#C87879", "#FACACA",
        "#E1E1E1", "#6D6D6D"
    ],
    N=len(t2m_bounds)
)
t2m_norm = BoundaryNorm(t2m_bounds, ncolors=len(t2m_bounds))

prec_bounds = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               12, 14, 16, 20, 24, 30, 40, 50, 60, 80, 100, 125]
prec_colors = ListedColormap([
    "#B4D7FF","#75BAFF","#349AFF","#0582FF","#0069D2",
    "#003680","#148F1B","#1ACF06","#64ED07","#FFF32B",
    "#E9DC01","#F06000","#FF7F26","#FFA66A","#F94E78",
    "#F71E53","#BE0000","#880000","#64007F","#C201FC",
    "#DD66FE","#EBA6FF","#F9E7FF","#D4D4D4","#969696"
])
prec_norm = BoundaryNorm(prec_bounds, prec_colors.N)

dbz_bounds = [0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 67, 70]
dbz_colors = ListedColormap([
    "#676767", "#FFFFFF", "#B3EFED", "#8CE7E2", "#00F5ED",
    "#00CEF0", "#01AFF4", "#028DF6", "#014FF7", "#0000F6",
    "#00FF01", "#01DF00", "#00D000", "#00BF00", "#00A701",
    "#019700", "#FFFF00", "#F9F000", "#EDD200", "#E7B500",
    "#FF5000", "#FF2801", "#F40000", "#EA0001", "#CC0000",
    "#FFC8FF", "#E9A1EA", "#D379D3", "#BE55BE", "#960E96"
])
dbz_norm = mcolors.BoundaryNorm(dbz_bounds, dbz_colors.N)

tp_acc_bounds = [0.1, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                 125, 150, 175, 200, 250, 300, 400, 500]
tp_acc_colors = ListedColormap([
    "#B4D7FF","#75BAFF","#349AFF","#0582FF","#0069D2",
    "#003680","#148F1B","#1ACF06","#64ED07","#FFF32B",
    "#E9DC01","#F06000","#FF7F26","#FFA66A","#F94E78",
    "#F71E53","#BE0000","#880000","#64007F","#C201FC",
    "#DD66FE","#EBA6FF","#F9E7FF","#D4D4D4","#969696"
])
tp_acc_norm = mcolors.BoundaryNorm(tp_acc_bounds, tp_acc_colors.N)

cape_bounds = [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
cape_colors = ListedColormap([
    "#676767", "#006400", "#008000", "#00CC00", "#66FF00", "#FFFF00", 
    "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000", "#FF0095", 
    "#FC439F", "#FF88D3", "#FF99FF"
])
cape_norm = mcolors.BoundaryNorm(cape_bounds, cape_colors.N)

pmsl_bounds_colors = list(range(912, 1070, 4))
pmsl_colors = LinearSegmentedColormap.from_list(
    "pmsl_smooth",
    [
       "#FF6DFF", "#C418C4", "#950CA2", "#5A007D", "#3D007F",
       "#00337E", "#0472CB", "#4FABF8", "#A3D4FF", "#79DAAD",
       "#07A220", "#3EC008", "#9EE002", "#F3FC01", "#F19806",
       "#F74F11", "#B81212", "#8C3234", "#C87879", "#F9CBCD",
       "#E2E2E2"
    ],
    N=len(pmsl_bounds_colors)
)
pmsl_norm = BoundaryNorm(pmsl_bounds_colors, ncolors=len(pmsl_bounds_colors))

# ------------------------------
# Windböen-Farben
# ------------------------------
wind_bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300]
wind_colors = ListedColormap([
    "#68AD05", "#8DC00B", "#B1D415", "#D5E81C", "#FBFC22",
    "#FAD024", "#F9A427", "#FC7929", "#FB4D2B", "#EA2B57",
    "#FB22A5", "#FC22CE", "#FC22F5", "#FC62F8", "#FD80F8",
    "#FFBFFC", "#FEDFFE", "#FEFFFF", "#E1E0FF", "#C3C3FF",
    "#A5A5FF", "#A5A5FF", "#6868FE"
])
wind_norm = mcolors.BoundaryNorm(wind_bounds, wind_colors.N)

# ------------------------------
# Kartenparameter
# ------------------------------
FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX
extent = [5, 16, 47, 56]

# ------------------------------
# WW-Legende Funktion
# ------------------------------
def add_ww_legend_bottom(fig, ww_categories, ww_colors_base):
    legend_height = 0.12
    legend_ax = fig.add_axes([0.05, 0.01, 0.9, legend_height])
    legend_ax.axis("off")
    for i, (label, codes) in enumerate(ww_categories.items()):
        n_colors = len(codes)
        block_width = 1.0 / len(ww_categories)
        gap = 0.05 * block_width
        x0 = i * block_width
        x1 = (i + 1) * block_width
        inner_width = x1 - x0 - gap
        color_width = inner_width / n_colors
        for j, c in enumerate(codes):
            color = ww_colors_base.get(c, "#FFFFFF")
            legend_ax.add_patch(mpatches.Rectangle((x0 + j * color_width, 0.3),
                                                  color_width, 0.6,
                                                  facecolor=color, edgecolor='black'))
        legend_ax.text((x0 + x1)/2, 0.05, label, ha='center', va='bottom', fontsize=10)

# ------------------------------
# ICON Grid laden (einmal!)
# ------------------------------
print(f"Lade ICON-Grid: {gridfile}")
nc = netCDF4.Dataset(gridfile)
lats = np.rad2deg(nc.variables["clat"][:]).flatten()
lons = np.rad2deg(nc.variables["clon"][:]).flatten()
nc.close()

# ------------------------------
# Dateien durchgehen
# ------------------------------
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".grib2"):
        continue
    path = os.path.join(data_dir, filename)
    ds = cfgrib.open_dataset(path)

    # --------------------------
    # Daten je Typ
    # --------------------------
    if var_type == "t2m":
        if "t2m" not in ds: continue
        data = ds["t2m"].values - 273.15
        cmap, norm = t2m_colors, t2m_norm
    elif var_type == "ww":
        varname = next((vn for vn in ds.data_vars if vn in ["WW","weather"]), None)
        if varname is None:
            print(f"Keine WW in {filename}")
            continue
        data = ds[varname].values
        cmap = None
    elif var_type == "tp_acc":
        if "tp" not in ds: continue
        data = ds["tp"].values
        data[data < 0.1] = 0 
        cmap, norm = tp_acc_colors, tp_acc_norm
        cmap.set_under('none')
    elif var_type == "pmsl":
        if "prmsl" not in ds:
            print(f"Keine prmsl-Variable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["prmsl"].values / 100
        data[data < 0] = np.nan
        cmap, norm = pmsl_colors, pmsl_norm
    elif var_type == "wind":
        if "fg10" not in ds:
            print(f"Keine passende Windvariable in {filename} ds.keys(): {list(ds.keys())}")
            continue
        data = ds["fg10"].values
        data[data < 0] = np.nan
        data = data * 3.6  # m/s → km/h
        cmap, norm = wind_colors, wind_norm
    else:
        print(f"Var_type {var_type} nicht implementiert")
        continue
    if data.ndim == 3:
        data = data[0].flatten()
    else:
        data = data.flatten()

    # --------------------------
    # Zeiten
    # --------------------------
    run_time_utc = pd.to_datetime(ds["time"].values) if "time" in ds else None
    if "valid_time" in ds:
        valid_time_raw = ds["valid_time"].values
        valid_time_utc = pd.to_datetime(valid_time_raw[0]) if np.ndim(valid_time_raw) > 0 else pd.to_datetime(valid_time_raw)
    else:
        step = pd.to_timedelta(ds["step"].values[0])
        valid_time_utc = run_time_utc + step
    valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))

    # --------------------------
    # Figure
    # --------------------------
    scale = 0.9
    fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)
    shift_up = 0.02
    ax = fig.add_axes([0.0, BOTTOM_AREA_PX/FIG_H_PX + shift_up, 1.0, TOP_AREA_PX/FIG_H_PX],
                      projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.set_axis_off()
    ax.set_aspect('auto')

    # ------------------------------
    # Regelmäßiges Gitter definieren
    # ------------------------------
    lon_min, lon_max, lat_min, lat_max = extent
    # Unterschiedliche Auflösung für WW
    if var_type == "ww":
        res = 0.15  # z.B. gröberes Raster für WW
    if var_type == "pmsl":
        res = 0.025
    else:
        res = 0.025  # Standard-Raster
    lon_grid = np.arange(lon_min, lon_max + res, res)
    lat_grid = np.arange(lat_min, lat_max + res, res)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # ------------------------------
    # Interpolation auf regelmäßiges Gitter
    # ------------------------------
    points = np.column_stack((lons, lats))
    valid_mask = np.isfinite(data)
    points_valid = points[valid_mask]
    data_valid = data[valid_mask]

    interpolator = NearestNDInterpolator(points_valid, data_valid)
    data_grid = interpolator(lon_grid, lat_grid)

    # --- Schnellere Raster-Darstellung statt tripcolor ---
    if cmap is not None:
        im = ax.pcolormesh(lon_grid, lat_grid, data_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
        if var_type == "t2m":
            smoothed_grid = gaussian_filter(data_grid, sigma=1.2)
            im = ax.pcolormesh(lon_grid, lat_grid, smoothed_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
            ax.contour(lon_grid, lat_grid, smoothed_grid, levels=t2m_bounds, colors='black', linewidths=0.3, alpha=0.5)
            n_labels = 40
            lon_min, lon_max, lat_min, lat_max = extent
            valid_mask = np.isfinite(data) & (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
            valid_indices = np.flatnonzero(valid_mask)
            np.random.shuffle(valid_indices)

            # KDTree für Städte vorbereiten
            city_coords = np.column_stack((cities['lon'], cities['lat']))
            city_tree = cKDTree(city_coords)

            min_city_dist = 1.0  # Mindestabstand in Grad
            texts = []
            used_points = 0
            tried_points = set()

            # Wir wählen n_labels zufällige Indizes aus valid_indices
            np.random.shuffle(valid_indices)

            while used_points < n_labels and len(tried_points) < len(valid_indices):
                idx = valid_indices[np.random.randint(0, len(valid_indices))]
                if idx in tried_points:
                    continue
                tried_points.add(idx)

                lon_pt, lat_pt = lons[idx], lats[idx]
                val = data[idx]

                # ---- schnelle Distanzprüfung via KDTree ----
                dist, _ = city_tree.query([lon_pt, lat_pt], distance_upper_bound=min_city_dist)
                if np.isfinite(dist):  # d.h. ein Stadtpunkt liegt innerhalb des Radius
                    continue
                # --------------------------------------------

                txt = ax.text(
                    lon_pt, lat_pt, f"{val:.0f}",
                    fontsize=10, ha='center', va='center', color='black', weight='bold'
                )
                txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
                texts.append(txt)
                used_points += 1

            adjust_text(texts, ax=ax, expand_text=(1.2, 1.2), arrowprops=None)
            
        elif var_type == "tp_acc":
            smoothed_grid = gaussian_filter(data_grid, sigma=1.2)
            im = ax.pcolormesh(lon_grid, lat_grid, smoothed_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")

        elif var_type == "wind":
            smoothed_grid = gaussian_filter(data_grid, sigma=1.2)
            im = ax.pcolormesh(lon_grid, lat_grid, smoothed_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
            smoothed_grid = gaussian_filter(data_grid, sigma=1.2)
            ax.contour(lon_grid, lat_grid, smoothed_grid, levels=wind_bounds, colors='black', linewidths=0.3, alpha=0.5)
            n_labels = 40
            lon_min, lon_max, lat_min, lat_max = extent
            valid_mask = np.isfinite(data) & (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
            valid_indices = np.flatnonzero(valid_mask)
            np.random.shuffle(valid_indices)

            # KDTree für Städte vorbereiten
            city_coords = np.column_stack((cities['lon'], cities['lat']))
            city_tree = cKDTree(city_coords)

            min_city_dist = 1.0  # Mindestabstand in Grad
            texts = []
            used_points = 0
            tried_points = set()

            # Wir wählen n_labels zufällige Indizes aus valid_indices
            np.random.shuffle(valid_indices)

            while used_points < n_labels and len(tried_points) < len(valid_indices):
                idx = valid_indices[np.random.randint(0, len(valid_indices))]
                if idx in tried_points:
                    continue
                tried_points.add(idx)

                lon_pt, lat_pt = lons[idx], lats[idx]
                val = data[idx]

                # ---- schnelle Distanzprüfung via KDTree ----
                dist, _ = city_tree.query([lon_pt, lat_pt], distance_upper_bound=min_city_dist)
                if np.isfinite(dist):  # d.h. ein Stadtpunkt liegt innerhalb des Radius
                    continue
                # --------------------------------------------

                txt = ax.text(
                    lon_pt, lat_pt, f"{val:.0f}",
                    fontsize=10, ha='center', va='center', color='black', weight='bold'
                )
                txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
                texts.append(txt)
                used_points += 1

            adjust_text(texts, ax=ax, expand_text=(1.2, 1.2), arrowprops=None)
        elif var_type == "pmsl":
            smoothed_grid = gaussian_filter(data_grid, sigma=1.2)
            im = ax.pcolormesh(lon_grid, lat_grid, smoothed_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
            data_hpa = smoothed_grid
            main_levels = list(range(912, 1070, 4))
            fine_levels = list(range(912, 1070, 2))
            main_levels = [lev for lev in main_levels if data_hpa.min() <= lev <= data_hpa.max()]
            fine_levels = [lev for lev in fine_levels if data_hpa.min() <= lev <= data_hpa.max()]

            cs_fine = ax.contour(lon_grid, lat_grid, data_hpa, levels=fine_levels,
                                colors='white', linewidths=0.3, alpha=0.8)
            cs_main = ax.contour(lon_grid, lat_grid, data_hpa, levels=main_levels,
                                colors='white', linewidths=1.2, alpha=1)

            used_points = set()
            texts = []
            min_city_dist = 1.1
            nlat, nlon = lon_grid.shape
            lat = lat_grid[:, 0]  # vertikale Achse
            lon = lon_grid[0, :]  # horizontale Achse
            lon_flat = lon_grid.flatten()
            lat_flat = lat_grid.flatten()

            def place_random_labels(cs, n_labels):
                contour_points = []
                for level_segs in cs.allsegs:
                    for seg in level_segs:
                        if seg.size > 0:
                            contour_points.extend(seg)
                contour_points = np.array(contour_points)

                lon_min, lon_max, lat_min, lat_max = extent
                mask = (contour_points[:,0] >= lon_min + 0.5) & (contour_points[:,0] <= lon_max - 0.5) & \
                       (contour_points[:,1] >= lat_min + 0.5) & (contour_points[:,1] <= lat_max - 0.5)
                contour_points = contour_points[mask]

                # Annahme: lon und lat sind die 1D-Achsen des Gitters
                lat_idx = np.searchsorted(lat, contour_points[:,1])
                lon_idx = np.searchsorted(lon, contour_points[:,0])

                # Clamping, um Indexfehler zu vermeiden
                lat_idx = np.clip(lat_idx, 0, len(lat) - 1)
                lon_idx = np.clip(lon_idx, 0, len(lon) - 1)

                # ij_points erstellen und Duplikate entfernen
                ij_points = np.column_stack((lat_idx, lon_idx))
                ij_points = np.unique(ij_points, axis=0)

                filtered_points = []
                for i, j in ij_points:
                    lon_pt, lat_pt = lon_grid[i, j], lat_grid[i, j]
                    if any(np.hypot(lon_pt - city_lon, lat_pt - city_lat) < min_city_dist
                           for city_lon, city_lat in zip(cities['lon'], cities['lat'])):
                        continue
                    if (i, j) in used_points:
                        continue
                    filtered_points.append((i, j))

                if len(filtered_points) > n_labels:
                    chosen_points = [filtered_points[i] for i in np.random.choice(len(filtered_points), n_labels, replace=False)]
                else:
                    chosen_points = filtered_points

                for i, j in chosen_points:
                    val = data_hpa[i, j]
                    txt = ax.text(lon_grid[i, j], lat_grid[i, j], f"{val:.0f}", fontsize=10,
                                 ha='center', va='center', color='black', weight='bold')
                    txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
                    texts.append(txt)
                    used_points.add((i, j))

            place_random_labels(cs_main, n_labels=5)
            place_random_labels(cs_fine, n_labels=5)

            # Extremwerte bestimmen
            min_val = np.min(data_hpa)
            max_val = np.max(data_hpa)

            # Positionen des minimalen und maximalen Werts finden
            min_pos = np.unravel_index(np.argmin(data_hpa), data_hpa.shape)
            max_pos = np.unravel_index(np.argmax(data_hpa), data_hpa.shape)

            # Tief (blau)
            txt_min = ax.text(lon[min_pos[1]], lat[min_pos[0]], f"{min_val:.0f}",
                            fontsize=14, color='blue', ha='center', va='center')
            txt_min.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])
            texts.append(txt_min)
            used_points.add(min_pos)

            # Hoch (rot)
            txt_max = ax.text(lon[max_pos[1]], lat[max_pos[0]], f"{max_val:.0f}",
                            fontsize=14, color='red', ha='center', va='center')
            txt_max.set_path_effects([path_effects.withStroke(linewidth=2, foreground="white")])
            texts.append(txt_max)
            used_points.add(max_pos)

            adjust_text(texts, ax=ax, expand_text=(1.5, 1.5), arrowprops=None)
    else:
        valid_mask = np.isfinite(data)
        codes = np.unique(data[valid_mask]).astype(int)
        codes = [c for c in codes if c in ww_colors_base]
        codes.sort()
        cmap = ListedColormap([ww_colors_base[c] for c in codes])
        code2idx = {c: i for i, c in enumerate(codes)}
        idx_data = np.full_like(data_grid, fill_value=np.nan, dtype=float)
        for c, i in code2idx.items():
            idx_data[data_grid == c] = i
        im = ax.pcolormesh(lon_grid, lat_grid, idx_data, cmap=cmap, vmin=-0.5, vmax=len(codes)-0.5, transform=ccrs.PlateCarree(), shading="auto")

    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
    for _, city in cities.iterrows():
        ax.plot(city["lon"], city["lat"], "o", markersize=6, markerfacecolor="black",
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        txt = ax.text(city["lon"]+0.1, city["lat"]+0.1, city["name"],
                      fontsize=9, color="black", weight="bold", zorder=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_patch(mpatches.Rectangle((0,0),1,1, transform=ax.transAxes, fill=False, color="black", linewidth=2))

    # --------------------------
    # Colorbar (falls relevant)
    # --------------------------
    legend_h_px = 50
    legend_bottom_px = 45
    if var_type in ["t2m", "dbz_cmax", "tp_acc", "cape_ml", "pmsl", "wind"]:
        bounds = t2m_bounds if var_type=="t2m" else tp_acc_bounds if var_type=="tp_acc" else cape_bounds if var_type=="cape_ml" else pmsl_bounds_colors if var_type=="pmsl" else wind_bounds
        cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
        cbar.ax.tick_params(colors="black", labelsize=7)
        cbar.outline.set_edgecolor("black")
        cbar.ax.set_facecolor("white")

        if var_type == "t2m":
            tick_labels = [str(tick) if tick % 4 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "pmsl":
            tick_labels = [str(tick) if tick % 8 == 0 else "" for tick in bounds]
            cbar.set_ticklabels(tick_labels)
        if var_type == "tp_acc":
            cbar.set_ticklabels([int(tick) if float(tick).is_integer() else tick for tick in tp_acc_bounds])
    else:
        add_ww_legend_bottom(fig, ww_categories, ww_colors_base)

    # Footer
    footer_ax = fig.add_axes([0.0, (legend_bottom_px + legend_h_px)/FIG_H_PX, 1.0,
                              (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px)/FIG_H_PX])
    footer_ax.axis("off")
    footer_texts = {
        "ww": "Signifikantes Wetter",
        "t2m": "Temperatur 2m (°C)",
        "tp_acc": "Akkumulierter Niederschlag (mm)",
        "pmsl": "Luftdruck auf Meereshöhe (hPa)",
        "wind": "Windböen (km/h)"
    }

    left_text = footer_texts.get(var_type, var_type) + \
                f"\nICON ({pd.to_datetime(run_time_utc).hour:02d}z), Deutscher Wetterdienst" \
                if run_time_utc is not None else \
                footer_texts.get(var_type, var_type) + "\nICON (??z), Deutscher Wetterdienst"

    footer_ax.text(0.01, 0.85, left_text, fontsize=12, fontweight="bold", va="top", ha="left")
    footer_ax.text(0.734, 0.92, "Prognose für:", fontsize=12, va="top", ha="left", fontweight="bold")
    footer_ax.text(0.99, 0.68, f"{valid_time_local:%d.%m.%Y %H:%M} Uhr",
                   fontsize=12, va="top", ha="right", fontweight="bold")

    outname = f"{var_type}_{valid_time_local:%Y%m%d_%H%M}.png"
    plt.savefig(os.path.join(output_dir, outname), dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()
