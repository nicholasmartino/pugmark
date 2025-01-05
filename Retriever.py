import gc
import io
import os
import warnings
import zipfile

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
from city.Fabric import Buildings, Parcels
from shapely.geometry import Polygon
from tqdm import tqdm

mpl.use("Agg")
warnings.simplefilter(action="ignore", category=FutureWarning)

CEILING_HEIGHT = 3


class Fabric:
    def __init__(self, parcels_gdf, buildings_gdf):
        self.parcels = Parcels(gdf=parcels_gdf)
        self.buildings = Buildings(gdf=buildings_gdf)


def update_footprints():
    # Download BC footprints from StatCan
    zipfile.ZipFile(
        io.BytesIO(
            requests.get(
                "https://www150.statcan.gc.ca/n1/fr/pub/34-26-0001/2018001/ODB_v2_BritishColumbia.zip?st=qdcH3z04"
            ).content
        )
    ).extractall("tmp/")
    print("Footprint data downloaded")


def update_parcels():
    # Download parcels from BC government open data
    zipfile.ZipFile(
        io.BytesIO(
            requests.get(
                "https://pub.data.gov.bc.ca/datasets/4cf233c2-f020-4f7a-9b87-1923252fbc24/pmbc_parcel_fabric_poly_svw.zip"
            ).content
        )
    ).extractall("tmp/")
    print("Parcel data downloaded")


def load_buildings(buildings_path):
    # Load building footprints from StatCan
    if not os.path.exists(buildings_path):
        update_footprints()
        os.makedirs(os.path.dirname(buildings_path), exist_ok=True)
        gdf = gpd.read_file("tmp/ODB_BritishColumbia/odb_britishcolumbia.shp")
        gdf.to_feather(buildings_path)

    # Load buildings and join height from BC Assessment
    buildings_gdf = gpd.read_feather(buildings_path).to_crs(26910)
    buildings_gdf["bid"] = buildings_gdf.reset_index(drop=True).index
    return buildings_gdf


def calculate_fsr(parcel_gdf, building_gdf):
    footprints = building_gdf.copy()
    overlay = gpd.overlay(parcel_gdf, footprints.loc[:, ["geometry"]])
    overlay["footprint_area"] = overlay.area

    # Compute the summed footprint_area for each 'id' in overlay
    summed_footprint = overlay.groupby("id")["footprint_area"].sum()

    # Map the summed values back to parcel_gdf
    parcel_gdf["footprint_area"] = parcel_gdf["id"].map(summed_footprint)

    storeys = "NUMBER_OF_STOREYS"

    # TODO: Guess missing number of storeys from footprint area and total building area (BCA)

    parcel_gdf.loc[parcel_gdf[storeys].isna(), storeys] = 2
    parcel_gdf.loc[parcel_gdf[storeys] == 0, storeys] = 2
    parcel_gdf["height"] = parcel_gdf[storeys] * CEILING_HEIGHT
    parcel_gdf["volume"] = parcel_gdf["footprint_area"] * parcel_gdf["height"]
    parcel_gdf["fsr"] = (parcel_gdf["volume"] / CEILING_HEIGHT) / parcel_gdf.area

    parcel_gdf.loc[
        :, ["id", "footprint_area", "height", "volume", "fsr", storeys, "geometry"]
    ].to_file("/Users/nicholasmartino/Desktop/parcel.geojson", driver="GeoJSON")
    footprints.loc[:, ["geometry"]].to_file(
        "/Users/nicholasmartino/Desktop/footprints.geojson", driver="GeoJSON"
    )
    overlay.to_file("/Users/nicholasmartino/Desktop/overlay.geojson", driver="GeoJSON")

    buildings = Buildings(gdf=building_gdf)
    buildings_centroids = buildings.gdf.copy()
    buildings_centroids["geometry"] = buildings_centroids.centroid.buffer(1)

    pcl = Parcels(gdf=parcel_gdf)
    pcl.gdf = pcl.gdf.drop_duplicates(subset=["geometry"])
    return pcl


def join_parcel_id(parcel_gdf, building_gdf):
    parcel_gdf = parcel_gdf.reset_index(drop=True)
    building_gdf = building_gdf.reset_index(drop=True)

    parcel_gdf["pid"] = parcel_gdf.index
    building_gdf["bid"] = building_gdf.index

    bld_gdf_ctr = building_gdf.copy()
    bld_gdf_ctr["geometry"] = bld_gdf_ctr.centroid.buffer(0.0001)

    # bld_gdf_join = bld_gdf_ctr.sjoin(parcels_gdf.loc[:, ['pid', 'geometry']])
    # grouped_join = bld_gdf_join.groupby('bid', as_index=False).first()

    bld_gdf_overlay = gpd.overlay(
        bld_gdf_ctr.loc[:, ["bid", "geometry"]], parcel_gdf.loc[:, ["pid", "geometry"]]
    )
    bld_gdf_overlay["area"] = bld_gdf_overlay.area
    grouped_overlay = (
        bld_gdf_overlay.sort_values("area", ascending=False)
        .groupby("bid", as_index=False)
        .first()
    )
    building_gdf.loc[grouped_overlay["bid"], "pid"] = list(grouped_overlay["pid"])
    return Fabric(parcel_gdf, building_gdf)


def translate_polygon(polygon, x_offset, y_offset):
    # Applies offset to each coordinate
    return Polygon([(x + x_offset, y + y_offset) for x, y in polygon.exterior.coords])


def plot_parcels(parcel_gdf, building_gdf, target_path):
    # Filter parcels by area and fsr
    processed_parcel_gdf = (
        parcel_gdf.loc[:, ["pid", "area", "volume", "fsr", "geometry"]].dropna().copy()
    )
    filtered_parcel_gdf = processed_parcel_gdf[
        (processed_parcel_gdf.area > 300)
        & (processed_parcel_gdf.fsr < 30)
        & (processed_parcel_gdf.area < 3000)
    ]

    # Plot parcel boundaries and building footprints
    parcel_boundary = filtered_parcel_gdf.copy()
    parcel_boundary["geometry"] = [geom for geom in parcel_boundary.boundary]
    parcel_boundary = parcel_boundary.set_geometry("geometry")

    # Make convex hull around largest parcel that will be plotted along with all other parcels to standardize the scale
    largest = (
        filtered_parcel_gdf.sort_values("area", ignore_index=True, ascending=False)
        .iloc[0]["geometry"]
        .convex_hull
    )
    largest_centroid = largest.centroid

    print("Plotting parcels and footprint skeletons")
    parcel_ids = filtered_parcel_gdf.pid

    # Get parcels not yet plotted
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)
    plotted = os.listdir(target_path)
    plotted_int = [int(i.split(".png")[0]) for i in plotted]
    not_plotted = set.difference(set(parcel_ids), set(plotted_int))

    for k, (j, t) in enumerate(zip(not_plotted, tqdm(range(len(not_plotted))))):
        j = int(j)

        # Move convex hull to parcel to standardize the plot scale
        p_centroid = filtered_parcel_gdf[filtered_parcel_gdf["pid"] == j].centroid
        x_offset = p_centroid.x - largest_centroid.x
        y_offset = p_centroid.y - largest_centroid.y
        largest_overlap = translate_polygon(largest, x_offset, y_offset)
        moved = gpd.GeoDataFrame({"geometry": [largest_overlap]}, geometry="geometry")

        # # Filter buildings with this parcel id
        # footprints = gpd.overlay(buildings.gdf, flt_parcels[flt_parcels['pid'] == j].loc[:, ['geometry']])

        footprints = building_gdf[building_gdf["pid"] == j].copy()

        if len(footprints) > 0:
            overlay = gpd.overlay(
                footprints, filtered_parcel_gdf[filtered_parcel_gdf["pid"] == j]
            )
            if len(overlay) > 0:
                parcel_boundary_color = "black"
                building_footprint_color = "gray"

                # Plot footprint, boundary and parcel
                fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

                moved.plot(ax=ax[0], color="white")
                moved.plot(ax=ax[1], color="white")

                filtered_parcel_gdf[filtered_parcel_gdf["pid"] == j].plot(
                    "fsr",
                    ax=ax[0],
                    cmap="viridis",
                    vmin=0,
                    vmax=5,
                    k=len(filtered_parcel_gdf),
                )
                filtered_parcel_gdf[filtered_parcel_gdf["pid"] == j].plot(
                    "fsr",
                    ax=ax[1],
                    cmap="viridis",
                    vmin=0,
                    vmax=5,
                    k=len(filtered_parcel_gdf),
                )

                parcel_boundary[parcel_boundary["pid"] == j].plot(
                    ax=ax[0], color=parcel_boundary_color
                )
                parcel_boundary[parcel_boundary["pid"] == j].plot(
                    ax=ax[1], color=parcel_boundary_color
                )

                footprints.plot(ax=ax[0], color=building_footprint_color)

                ax[0].set_axis_off()
                ax[1].set_axis_off()

                fig.savefig(fname=f"{target_path}/{j}.png", dpi=64)
                plt.close()

        gc.collect()
    return


def export_parcels(parcel: Parcels, directory: str):
    parcel.gdf.to_feather(f"{directory}/parcel_far.feather")
    parcel.gdf.loc[
        ~parcel.gdf["fsr"].isna() & parcel.gdf["fsr"] > 0,
        ["pid", "ACTUAL_TOTAL", "ACTUAL_LAND", "fsr", "geometry"],
    ].to_crs(4326).to_file(f"{directory}/parcel_far.geojson", driver="GeoJSON")
    return


if __name__ == "__main__":
    data_dir = "data/Metro Vancouver Regional District"
    building_source_path = f"{data_dir}/statistics_canada/building_footprints.feather"
    plot_target_dir = f"{data_dir}/processed/footprints"

    buildings = load_buildings(building_source_path)
    parcels = gpd.read_feather(f"{data_dir}/bc_assessment/parcel.feather")

    processed_parcels = calculate_fsr(parcels, buildings)
    export_parcels(processed_parcels, f"{data_dir}/processed/samples")

    fabric = join_parcel_id(processed_parcels.gdf, buildings)
    plot_parcels(fabric.parcels.gdf, fabric.buildings.gdf, plot_target_dir)
