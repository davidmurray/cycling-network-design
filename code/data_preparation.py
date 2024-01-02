import pathlib
import geopandas as gpd
import osmnx as ox
from masters import *
import argparse

parser = argparse.ArgumentParser(description="Bicycle network design data preparation")
parser.add_argument("--zone_polygon", dest="zone_polygon_path", required=True, help="Path to the GeoJSON file containing the area of the zone")
parser.add_argument("--od_data", dest="od_data_path", required=True, help="Path to the csv file containing OD demand data")
parser.add_argument("--output_directory", required=True, help="Path to directory where to save files")
parser.add_argument("--local_crs", default="epsg:2950", help="EPSG code for the local coordinate system (default: epsg:2950)")
parser.add_argument("--minimum_euclidean_distance", default="epsg:2950", help="Minimum euclidean distance of trips in order to be retained (default: 1000 meters)", type=int)
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode (default: False)")
args = parser.parse_args()

print("Starting data preparation for args", args.zone_polygon_path, "saving data to", args.output_directory)

output_directory = pathlib.Path(args.output_directory)
output_directory.mkdir(parents=True, exist_ok=True)
od_data_path = pathlib.Path(args.od_data_path)

zone_gdf = gpd.read_file(args.zone_polygon_path)
zone_polygon = zone_gdf.geometry.iloc[0]
ox.settings.log_console = args.verbose
if "all_private" not in ox.settings.bidirectional_network_types:
    ox.settings.bidirectional_network_types.append('all_private')

G = ox.graph_from_polygon(zone_polygon, 
                          network_type="all_private",
                          clean_periphery=True,
                          simplify=False, # Do not simplify the graph yet as we do it latter in a slightly more sophisticated way (intersection grouping then simplifying)
                          retain_all=True,
                          truncate_by_edge=True)
G = ox.project_graph(G, to_crs=args.local_crs)

od_df = load_od_csv(od_data_path)
od_df_filtered = filter_od_csv(od_df, zone_polygon, min_euclidean_distance=args.minimum_euclidean_distance, verbose=args.verbose)
od_df_filtered.to_csv(output_directory / "od_df_filtered.csv", index=False)

G_consolidated = consolidate_graph(G)
ox.save_graphml(G_consolidated, filepath=output_directory / "G_consolidated.graphml")

G_simplified = simplify_consolidated_graph(G_consolidated)
for i, (u, v, k) in enumerate(G_simplified.edges(keys=True)):
    G_simplified[u][v][k]['sequential_id'] = i
ox.save_graphml(G_simplified, filepath=output_directory / "G_simplified.graphml")

save_network_for_osrm(G_simplified, path=output_directory / "osrm_network.xml")
