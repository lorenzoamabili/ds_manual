# 22 · Geospatial Analysis

When *location* is a first-class variable: logistics, retail siting, real estate,
mobility, agriculture, climate, epidemiology, telecoms. Geospatial data breaks
several assumptions of ordinary ML and rewards its own toolkit.

## Two data models
- **Vector** — points (stores, sensors), lines (roads, rivers), polygons (regions,
  buildings). Stored as geometries with a coordinate reference system.
- **Raster** — a grid of cells (satellite imagery, elevation, temperature fields).
  Analysed with array/CV techniques ([11](11-computer-vision.md)).

**Coordinate reference systems (CRS) are the classic footgun.** Latitude/longitude
(EPSG:4326) are degrees, not metres — computing distances or areas on them directly
is wrong. Reproject to an appropriate projected CRS (e.g. a local UTM zone) before
any metric operation. More geospatial bugs come from CRS confusion than anything
else.

## The spatial operations you'll actually use
- **Spatial joins** — attach attributes by location ("which region contains this
  point", "which stores are within 1 km").
- **Buffers, intersections, unions, dissolves** — the geometric algebra of
  catchment areas, overlaps, and aggregation.
- **Distance & nearest-neighbour** — but on the road/network distance when that's
  what matters, not straight-line (a river between two nearby points).
- **Spatial indexing** (R-tree, H3/S2 hex grids) — makes "find things near X"
  tractable at scale; H3 hexagons are a popular way to bin and aggregate point data.

## Why ordinary ML assumptions break
- **Spatial autocorrelation** — "everything is related to everything, but near things
  more so" (Tobler's first law). Nearby observations are *not* independent, so naive
  standard errors are too optimistic. Measure it (Moran's I) and account for it.
- **Spatial [leakage](03-data-and-feature-engineering.md)** — a random train/test split puts neighbouring, correlated
  points on both sides, inflating scores. Use **spatial [cross-validation](04-evaluation-and-validation.md)** (hold out
  whole regions/blocks), the geospatial analogue of [time-series](07-time-series-forecasting.md) CV.
- **Modifiable Areal Unit Problem (MAUP)** — results change with the choice of
  boundaries/aggregation level; conclusions can be an artefact of the zoning.

## Methods
- **Spatial statistics** — hotspot analysis (Getis-Ord Gi*), spatial
  autocorrelation (Moran's I), and **kriging** (geostatistical interpolation with
  uncertainty) for filling in a field from sampled points.
- **Spatial regression** — spatial lag / spatial error models, or geographically
  weighted regression (GWR) for effects that vary across space.
- **ML with spatial features** — coordinates, distances-to-features, neighbourhood
  aggregates, H3-cell statistics; [gradient boosting](05-supervised-learning.md) on these is a strong practical
  baseline. For imagery, CNNs on raster tiles.
- **Routing & optimisation** — shortest path, isochrones, and [vehicle routing](12-optimization.md) sit at
  the border with [optimisation](12-optimization.md).

## Tools
`GeoPandas` (vector analysis, pandas-like), `Shapely` (geometry), `rasterio`/`xarray`
(rasters), `pyproj` (CRS), `folium`/`kepler.gl`/`pydeck` (interactive maps), `H3`
(hex indexing), `OSMnx` (street networks), and `PostGIS` (spatial SQL in Postgres)
when location lives in the warehouse.

## Practical checklist
Confirm the CRS before any distance/area calc; visualise the raw geometries early
(bad joins are obvious on a map); use spatial CV; and decide the aggregation unit
deliberately, aware of MAUP. A quick map catches errors that no summary statistic
will.

---

## Python example — spatial join, H3 aggregation, and spatial autocorrelation

```python
"""
Geospatial analysis:
  1. Point-in-polygon spatial join
  2. H3 hexagonal binning of point events
  3. Nearest-neighbour distance (vectorised, no CRS error)
  4. Moran's I autocorrelation demo

Uses synthetic data — no external files required.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

rng = np.random.default_rng(42)

# ── Synthetic city: 1000 incidents in a 10km x 10km grid ─────────────────────
n = 1_000
# Two clusters (hotspots) + background noise
cluster1 = rng.normal([2, 3], 0.5, (400, 2))
cluster2 = rng.normal([7, 7], 0.6, (300, 2))
background = rng.uniform(0, 10, (300, 2))
coords = np.vstack([cluster1, cluster2, background])
incidents = pd.DataFrame(coords, columns=["x","y"])

# ── 1. Point-in-polygon: assign district ─────────────────────────────────────
# 4 districts: quadrants of the 10x10 grid
def assign_district(row):
    return ("NW" if row.x < 5 and row.y >= 5 else
            "NE" if row.x >= 5 and row.y >= 5 else
            "SW" if row.x < 5 else "SE")
incidents["district"] = incidents.apply(assign_district, axis=1)
print("Incident count by district:")
print(incidents["district"].value_counts().to_string())

# ── 2. H3-style hexagonal binning (approximated with a hex grid) ─────────────
# Bin into 0.5x0.5 cells and count
cell_size = 1.0
incidents["cell_x"] = (incidents["x"] / cell_size).astype(int)
incidents["cell_y"] = (incidents["y"] / cell_size).astype(int)
hotspots = incidents.groupby(["cell_x","cell_y"]).size().reset_index(name="count")
top_cells = hotspots.nlargest(5, "count")
print(f"\nTop 5 hotspot cells (H3-style binning, cell_size={cell_size}km):")
print(top_cells.to_string(index=False))

# ── 3. Nearest-neighbour distance (nearest police station) ───────────────────
stations = pd.DataFrame({"x":[1,5,9,3,7], "y":[1,5,9,8,2]})
tree = cKDTree(stations[["x","y"]].values)
distances, _ = tree.query(incidents[["x","y"]].values)
incidents["dist_to_station"] = distances

print(f"\nMean distance to nearest station: {distances.mean():.2f} km")
print(f"Max distance (coverage gap):     {distances.max():.2f} km")

# ── 4. Spatial autocorrelation: grid counts ───────────────────────────────────
# Moran's I ≈ 1: strong spatial clustering; ≈ 0: random; ≈ -1: dispersed
grid_counts = hotspots.set_index(["cell_x","cell_y"])["count"].values
n_cells = len(grid_counts)
mean_c = grid_counts.mean()
z = grid_counts - mean_c

# Simplified Moran's I (queen contiguity approximated as nearest-cell in 1D for demo)
ss = (z**2).sum()
lag_z = np.roll(z, 1)  # simplified 1-step lag
morans_i = n_cells * (z * lag_z).sum() / (n_cells * ss + 1e-9)
print(f"\nSimplified Moran's I: {morans_i:.3f}  (positive = spatial clustering)")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sc = ax1.scatter(incidents["x"], incidents["y"], c=incidents["dist_to_station"],
                 cmap="Reds_r", s=8, alpha=0.7)
ax1.scatter(stations["x"], stations["y"], marker="*", s=200, c="blue",
            zorder=5, label="Police stations")
plt.colorbar(sc, ax=ax1, label="Distance to nearest station (km)")
ax1.set_title("Incidents coloured by coverage distance")
ax1.legend()

pivot = hotspots.pivot_table(index="cell_y", columns="cell_x", values="count", fill_value=0)
ax2.imshow(pivot.values, origin="lower", cmap="hot_r", aspect="auto")
ax2.set_title("Incident density heatmap\n(hex-bin approximation)")
ax2.set_xlabel("x cell"); ax2.set_ylabel("y cell")
plt.colorbar(ax2.images[0], ax=ax2, label="Count")

plt.tight_layout()
plt.savefig("geospatial_analysis.png", dpi=120)
plt.close()
print("\nPlot saved. Two clear hotspots visible in heatmap.")

# ── CRS warning ───────────────────────────────────────────────────────────────
print("\nReminder: always confirm CRS before distance/area calculations.")
print("Computing distances on lat/lon degrees gives wrong answers — reproject to UTM.")
```

---

## Cross-references

- [34](34-manufacturing.md) — IoT sensor spatial analysis
- [32](32-retail-ecommerce.md) — retail site selection and catchment analysis
- [12](12-optimization.md) — routing and facility location
