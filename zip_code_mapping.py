import geopandas as gpd
import fiona


# Locate data
# drive.mount('/content/drive')
# os.chdir("drive/.shortcut-targets-by-id/1UCSGslwBeg2VrVqVpXdGkVODXRXfYIzf/MADS Milestone 1")

gdb_path = "./data/WalkabilityIndex/Natl_WI.gdb"

#List available layers inside the geodatabase
layers = fiona.listlayers(gdb_path)
print("Layers found:", layers)

gdf = gpd.read_file(gdb_path, layer="NationalWalkabilityIndex")

zcta = gpd.read_file("data/tl_2020_us_zcta520/tl_2020_us_zcta520.shp")

def get_zip_block_mapping(zip_codes):

  def calc_avg(row):
    geo_ids = row["GEOID"]
    isolated_blocks = gdf[gdf["GEOID10"].isin(geo_ids)]
    return isolated_blocks["NatWalkInd"].mean()

  zcta_df = zcta[zcta["ZCTA5CE20"].isin(zip_codes)]

  # intersection
  walk_blocks = gdf.to_crs(zcta_df.crs)
  zip_union = zcta_df.unary_union
  intersected_blocks_df = walk_blocks[walk_blocks.intersects(zip_union)]

  # are calculation
  intersetion_df = gpd.overlay(intersected_blocks_df, zcta_df, how="intersection")
  intersetion_df["overlap_m2"] = intersetion_df.geometry.area

  pivot = intersetion_df.pivot_table(
    index="GEOID10",
    columns="ZCTA5CE20",
    values="overlap_m2",
    fill_value=0
  )

  pivot_block_max = pivot.idxmax(axis=1)

  relation_df = pivot_block_max.to_frame().reset_index()
  relation_df.columns = ["GEOID", "ZIP"]
  zip_block_mapping = relation_df.groupby("ZIP").agg(list)
  zip_block_mapping.head()

  zip_block_mapping["NatWalkIndAvg"] = zip_block_mapping.apply(calc_avg, axis=1)

  return zip_block_mapping