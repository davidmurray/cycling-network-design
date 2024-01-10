import os
import time
import logging
import copy
import random
import json
import ujson
import types
import pathlib
import subprocess
import shutil
from glob import glob
import pandas as pd
import numpy as np
import pyproj
import shapely
import networkx as nx
import osmnx as ox
import folium
import itertools
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#from hilbertcurve.hilbertcurve import HilbertCurve

## Plotting utilities

def random_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)

def create_map(tiles="cartodbpositron", **kwargs):
    return folium.Map(location=(45.5, -73.6), zoom_start=12, tiles=tiles, preferCanvas=True, max_zoom=19, **kwargs)

def plot_network(network_df, map_=None, color=None, popup=["name"], tile="cartodbpositron", weight=3):
    if not map_:
        map_ = create_map(tiles=tile)

    if not color:
        colors = [random_color() for _ in range(len(network_df))]
    elif type(color) != list:
        colors = itertools.repeat(color)
    else:
        colors = color
        
    for (i, row), color in zip(network_df.iterrows(), colors):
        locations = [(lat, lng) for lng, lat in row.geometry_wgs84.coords]
        #polyline = folium.PolyLine(locations=locations, popup=" ".join([str(row[attr]) for attr in popup]), color=color, weight=weight)
        polyline = folium.PolyLine(locations=locations, popup=row.name, color=color, weight=weight)
        polyline.add_to(map_)
    del network_df
    
    return map_

## Shapely utility
from shapely.ops import transform

def round_coordinates(geom, ndigits=4):
    def _round_coords(x, y, z=None):
        x = round(x, ndigits)
        y = round(y, ndigits)

        if z is not None:
            z = round(x, ndigits)

        return [c for c in (x, y, z) if c is not None]

    return transform(_round_coords, geom)

## Origin-Destination

def load_od_csv(path):
    od_df = pd.read_csv(path)

    proj = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:4326'), # source coordinate system
        pyproj.Proj('epsg:2950')) # destination coordinate system: https://epsg.io/32618

    od_df['origin_wgs84'] = od_df.apply(lambda row: shapely.geometry.Point(row.lonorig, row.latorig), axis=1)
    od_df['destination_wgs84'] = od_df.apply(lambda row: shapely.geometry.Point(row.londest, row.latdest), axis=1)

    od_df['origin'] = od_df.apply(lambda row: shapely.ops.transform(proj.transform, shapely.geometry.Point(row.latorig, row.lonorig)), axis=1)
    od_df['destination'] = od_df.apply(lambda row: shapely.ops.transform(proj.transform, shapely.geometry.Point(row.latdest, row.londest)), axis=1)
    od_df['euclidean_distance'] = od_df.apply(lambda row: row.origin.distance(row.destination), axis=1)

    return od_df

def filter_od_csv(od, area_polygon, min_euclidean_distance=0, max_euclidean_distance=np.inf, verbose=False):
    # For now we will restrict to trips that start and end on the island of Montreal
    len_before = len(od)

    #valid_points = od_df.origin_wgs84.apply(lambda x: x.within(boundary_polygon)) & od_df.destination_wgs84.apply(lambda x: x.within(boundary_polygon))
    valid_points = od.origin_wgs84.apply(lambda x: x.within(area_polygon)) & od.destination_wgs84.apply(lambda x: x.within(area_polygon))
    od = od[valid_points]
    od = od[(od.euclidean_distance >= min_euclidean_distance) & (od.euclidean_distance <= max_euclidean_distance)]
    if verbose:
        print(f"Dropped {len_before-len(od)} ({(len_before-len(od))/len_before * 100:0.1f} %) trips that are too short or don't fall within the polygon boundary. Remaining trips: {len(od)}")
    return od
    # FIXME : od_df contains sometimes a trip and its reverse! Is this valuable info? Can we discard one of them?

def create_od_df_for_zone_partition(od_dataframe, partition): # boundary is a geodataframe
    from shapely.geometry import LineString, Point
    from shapely.ops import transform

    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:2950'), # source coordinate system
        pyproj.Proj('epsg:4326')) # destination coordinate system

    partition = partition.to_crs("epsg:2950")
    zone_geom = partition.geometry.iloc[0]
    zone_trips = []

    for _, row in od_dataframe.iterrows():
        od_line = LineString([row.origin, row.destination])

        intersection = zone_geom.intersection(od_line)
        if intersection.is_empty:
            continue

        if isinstance(intersection, shapely.geometry.MultiLineString):
            lines = intersection.geoms
        else:
            lines = [intersection]
        for line in lines:
            zone_trips.append({'origin': round_coordinates(Point(line.coords[0])), 'destination': round_coordinates(Point(line.coords[1]))})

    zone_trips_df = pd.DataFrame(zone_trips)
    
    x_orig = zone_trips_df.apply(lambda row: row.origin.x, axis=1)
    y_orig = zone_trips_df.apply(lambda row: row.origin.y, axis=1)
    x_dest = zone_trips_df.apply(lambda row: row.destination.x, axis=1)
    y_dest = zone_trips_df.apply(lambda row: row.destination.y, axis=1)
    transformer = pyproj.Transformer.from_crs("EPSG:2950", "EPSG:4326")
    latorig, lonorig = transformer.transform(x_orig, y_orig)
    latdest, londest = transformer.transform(x_dest, y_dest)
    zone_trips_df['latorig'] = latorig
    zone_trips_df['lonorig'] = lonorig
    zone_trips_df['latdest'] = latdest
    zone_trips_df['londest'] = londest
    zone_trips_df['origin_wgs84'] = [round_coordinates(Point(lon, lat), 7) for lon, lat in zip(lonorig, latorig)]
    zone_trips_df['destination_wgs84'] = [round_coordinates(Point(lon, lat), 7) for lon, lat in zip(londest, latdest)]
    
    return zone_trips_df
    
def create_exploded_od_df(od_df, edges_df, buffer_radius=500):
    for column in ['origin', 'destination']:
        try:
            od_df[column] = shapely.wkt.loads(od_df[column])
        except TypeError: # Column was probably already in shapely format. We can ignore this error.
            pass

    od_df_exploded_data = []
    spatial_index = edges_df.geometry.sindex

    for idx, row in od_df.iterrows():
        def find_start_points(center):
            polygon = center.buffer(buffer_radius)

            possible_matches_index = spatial_index.query(polygon)
            possible_matches = edges_df.iloc[possible_matches_index]
            intersecting = possible_matches[possible_matches.intersects(polygon)]

            pts = {}
            for idx, row in intersecting.iterrows():
                geom = row.geometry
                if geom.is_empty:
                    continue
                pts[idx] = round_coordinates(shapely.ops.nearest_points(geom, center)[0])
            return pts

        orig_start_points = find_start_points(row.origin)
        dest_start_points = find_start_points(row.destination)
        orig_start_points_inv = {v: k for k, v in orig_start_points.items()}
        dest_start_points_inv = {v: k for k, v in dest_start_points.items()}

        for pairs in itertools.product(orig_start_points.values(), dest_start_points.values()):
            od_df_exploded_data.append({'od_df_idx': idx,
                              'origin': round_coordinates(row.origin),
                              'destination': round_coordinates(row.destination),
                              'origin_disag': pairs[0],
                              'origin_dist': round(row.origin.distance(pairs[0]), 1),
                              'destination_disag': pairs[1],
                              'destination_dist': round(row.destination.distance(pairs[1]), 1),
                              'origin_link': orig_start_points_inv[pairs[0]],
                              'destination_link': dest_start_points_inv[pairs[1]]
            })

    od_df_exploded = pd.DataFrame(od_df_exploded_data)
    od_df_exploded['orig_dest_tot_disag_dist'] = od_df_exploded.origin_dist + od_df_exploded.destination_dist
    
    x_orig = od_df_exploded.apply(lambda row: row.origin_disag.x, axis=1)
    y_orig = od_df_exploded.apply(lambda row: row.origin_disag.y, axis=1)
    x_dest = od_df_exploded.apply(lambda row: row.destination_disag.x, axis=1)
    y_dest = od_df_exploded.apply(lambda row: row.destination_disag.y, axis=1)
    transformer = pyproj.Transformer.from_crs("EPSG:2950", "EPSG:4326")
    latorig, lonorig = transformer.transform(x_orig, y_orig)
    latdest, londest = transformer.transform(x_dest, y_dest)
    od_df_exploded['latorig'] = latorig
    od_df_exploded['lonorig'] = lonorig
    od_df_exploded['latdest'] = latdest
    od_df_exploded['londest'] = londest
    od_df_exploded = od_df_exploded.drop_duplicates() # Remove duplicates that sometimes arise due to edges sharing common nodes at a cul-de-sac, for example
    od_df_exploded.attrs = {'len_od_df': len(od_df), 'buffer_radius': buffer_radius}
    
    return od_df_exploded

def filter_exploded_od_df(gdf_net, G, exploded_od_df):
    initial_attrs = exploded_od_df.attrs
    # Create a dataframe for all the connected components in the network.
    network_graph = G.edge_subgraph(gdf_net.index)
    components_df = []
    for i, component_nodes in enumerate(nx.connected_components(network_graph)):
        component = network_graph.subgraph(component_nodes)
        for edge in component.edges:
            # Since this is an undirected graph, the subgraph function
            # will sometimes give edges that are in a different order than in the index above.
            # If that's the case, we just manually fix the order. A bit of a hack...
            if edge not in gdf_net.index:
                edge = (edge[1], edge[0], edge[2])
            components_df.append({'component_id': i, 'edge': edge})
    components_df = pd.DataFrame(components_df)

    # Remove all possible paths that start or end on an edge that isn't part of the network.
    exploded_od_df = exploded_od_df[(exploded_od_df.origin_link.isin(gdf_net.index)) & (exploded_od_df.destination_link.isin(gdf_net.index))]

    # Get the component that each edge is on
    tmp = pd.merge(exploded_od_df, components_df, left_on='origin_link', right_on='edge').rename(columns={'component_id': 'origin_component_id'})
    exploded_od_df = pd.merge(tmp, components_df, left_on='destination_link', right_on='edge').rename(columns={'component_id': 'destination_component_id'})
    del tmp

    del exploded_od_df['edge_x']
    del exploded_od_df['edge_y']

    # Only retain trips that are on the same component since others will obviously not be routable.
    exploded_od_df = exploded_od_df[exploded_od_df.origin_component_id == exploded_od_df.destination_component_id]

    exploded_od_df.attrs = initial_attrs
    return exploded_od_df

## OpenStreetMap graph utilities
def load_osm_graph(path):
    ox.settings.log_console = True
    if "all_private" not in ox.settings.bidirectional_network_types:
        ox.settings.bidirectional_network_types.append('all_private')

    G = ox.graph_from_xml(path,
                          simplify=False, # Do not simplify the graph yet as we do it later in a slightly more sophisticated way (intersection grouping then simplifying)
                          retain_all=True,
                          bidirectional=True) # We want bidirectional edges, even for one-way streets.
    return G


def add_wgs84_column_to_df(gdf):
    # Make a backup of the geometry in EPSG:2950 foramat
    gdf['geometry_2950'] = gdf['geometry']
    gdf = gdf.to_crs('epsg:4326')
    gdf = gdf.rename(columns={'geometry': 'geometry_wgs84'})
    gdf = gdf.rename(columns={'geometry_2950': 'geometry'})
    return gdf

def consolidate_graph(G,
                      tolerance=11, # Nodes buffering radius
                      allowed_highway_types=["primary",
                             #"primary_link",
                             "secondary",
                             #"secondary_link",
                             "tertiary",
                             #"tertiary_link"
                             #"residential", # Why did I include those?
                             #"cycleway" # Why did I include those?
                            ]):

    edges_subset = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if data['highway'] in allowed_highway_types:
            edges_subset.append((u, v, key))

    G_subset = G.edge_subgraph(edges_subset)
    G_consolidated = ox.consolidate_intersections(G_subset,
                                                  rebuild_graph=True, # We want to get a networkx graph back, not just a list of consolidated intersection nodes.
                                                  tolerance=tolerance, # Nodes buffering radius
                                                  dead_ends=False # Do not discard dead ends
                                                 )
    return G_consolidated

def simplify_consolidated_graph(consolidated_graph):
    out_data = dict()
    
    ## Step 1: Retain only edges of a certain type
    # For the genetic algorithm, we only keep these types of edges:
    allowed_highway_types = ["primary",
                             "secondary",
                             "tertiary"]
    edges_subset = []
    for u, v, key, data in consolidated_graph.edges(keys=True, data=True):
        if data['highway'] in allowed_highway_types:
            edges_subset.append((u, v, key))
    subset_G = consolidated_graph.edge_subgraph(edges_subset).copy()

    ## Step 2: In case of parallel edges, keep only a single edge.
    to_remove = []
    # identify all the parallel edges in the MultiDiGraph
    parallels = ((u, v) for u, v in subset_G.edges(keys=False) if subset_G.number_of_edges(u, v) > 1)
    # among all sets of parallel edges, remove all except the one with the minimum length
    for u, v in set(parallels):
        k_min, _ = min(subset_G.get_edge_data(u, v).items(), key=lambda x: x[1]["length"])
        to_remove.extend((u, v, k) for k in subset_G[u][v] if k != k_min)
    subset_G.remove_edges_from(to_remove)

    out_data['step2'] = subset_G.copy()
    
    ## Step 3: Simplify graph
    simplified_G = ox.simplify_graph(subset_G,
                                     strict=True,
                                     remove_rings=False,
                                     track_merged=True)
    out_data['step3'] = simplified_G.copy()

    ## Step 4: Add missing merged_edges attribute
    # For every edge that was not merged with other edges,
    # we set the "merged_edges" property to the edge itself.
    # This is necessary when further simplifying edges
    for u, v, k, data in simplified_G.edges(keys=True, data=True):
        if 'merged_edges' not in data:
            simplified_G[u][v][k]['merged_edges'] = [(u, v)]
        #G_simplified[u][v][k]['merged_edges'] = [(99, 9999), (88, 8888)]

    #gdf_nodes_G_simplified, gdf_links_G_simplified = ox.graph_to_gdfs(simplified_G)
    #gdf_nodes_G_simplified = add_wgs84_column_to_df(gdf_nodes_G_simplified)
    #gdf_links_G_simplified = add_wgs84_column_to_df(gdf_links_G_simplified)

    ## Step 5: after simplifying, remove parallel edges once again
    pruned_G = simplified_G.copy()
    to_remove = []

    # identify all the parallel edges in the MultiDiGraph
    parallels = ((u, v) for u, v in pruned_G.edges(keys=False) if pruned_G.number_of_edges(u, v) > 1)

    # among all sets of parallel edges, remove all except the one with the minimum length
    for u, v in set(parallels):
        k_min, _ = min(pruned_G.get_edge_data(u, v).items(), key=lambda x: x[1]["length"])
        to_remove.extend((u, v, k) for k in pruned_G[u][v] if k != k_min)

    pruned_G.remove_edges_from(to_remove)
    out_data['step5'] = pruned_G.copy()


    ## Step 6: After removing parallel edges, simplify a final time

    #import importlib; importlib.reload(sys.modules['graph_simplification'])
    from graph_simplification import simplify_graph, _is_endpoint, _get_paths_to_simplify, _merge_nodes_geometric

    # Allow simplification again
    pruned_G.graph['simplified'] = False
    simplified_G = simplify_graph(pruned_G,
                                   strict=True,
                                   remove_rings=False,
                                   track_merged=False) # track_merged is False because we just want to combine the merged_edges attribute of every edge that we merge.

    # Convert to an undirected graph. We don't want parallel edges unless their geometries differ.
    simplified_G = ox.get_undirected(simplified_G)
    out_data['step6'] = simplified_G.copy()

    del pruned_G

    return simplified_G#, out_data

def save_network_for_osrm(consolidated_graph, path):
    # Save the OSM network as an XML file for OSRM

    gdf_nodes_G, gdf_links_G = ox.graph_to_gdfs(consolidated_graph)
    gdf_links_G = add_wgs84_column_to_df(gdf_links_G)
    gdf_nodes_G = add_wgs84_column_to_df(gdf_nodes_G)

    gdf_nodes_c = gdf_nodes_G.copy()
    gdf_links_c = gdf_links_G.copy()

    # TODO: Investigate why
    # Some nodes have null lon & lat, so we set them to the correct value for all using the geometry column.
    gdf_nodes_c.lon = gdf_nodes_c.geometry_wgs84.x
    gdf_nodes_c.lat = gdf_nodes_c.geometry_wgs84.y
    gdf_nodes_c.x = gdf_nodes_c.geometry_wgs84.x
    gdf_nodes_c.y = gdf_nodes_c.geometry_wgs84.y

    # Save the consolidated network to an OSM XML file
    ox.save_graph_xml((gdf_nodes_c, gdf_links_c),
                      filepath=path,
                      node_tags=['highway'],
                      node_attrs=['id', 'lat', 'lon'],
                      edge_tags=['highway', 'lanes', 'maxspeed', 'name', 'oneway', 'sequential_id'],
                      edge_attrs=['id'])

    del gdf_nodes_c
    del gdf_links_c

## Genetic algorithm
"""
def sort_using_hilbert_curve(edges_df, n, p):
    hilbert_curve = HilbertCurve(p, n)

    #edges_df['midpoint_wgs84'] = edges_df.apply(lambda row: row.geometry_wgs84.interpolate(0.5, normalized = True), axis=1) # This gets the center point on each edge.
    edges_df['midpoint'] = edges_df.apply(lambda row: row.geometry.interpolate(0.5, normalized=True), axis=1) # This gets the center point on each edge.

    # Split into x and y columns and then substract minimum to start at 0.
    edges_df['midpoint_x'] = edges_df.apply(lambda row: row.midpoint.coords[0][0], axis=1)
    edges_df['midpoint_y'] = edges_df.apply(lambda row: row.midpoint.coords[0][1], axis=1)
    edges_df['midpoint_x'] = edges_df['midpoint_x'] - edges_df['midpoint_x'].min()
    edges_df['midpoint_y'] = edges_df['midpoint_y'] - edges_df['midpoint_y'].min()

    # Calculate point along Hilbert curve
    edges_df['hilbert_index'] = edges_df.apply(lambda row: hilbert_curve.distance_from_point((row.midpoint_x, row.midpoint_y)), axis=1)

    # This is the key part: sort by position along Hilbert curve.
    edges_df_sorted = edges_df.sort_values('hilbert_index')

    # Change to have a range from 0 to N, where N is the number of links.
    edges_df_sorted['hilbert_index_linear'] = range(0, len(edges_df))

    return edges_df_sorted
"""

def calculate_network_length(network):
    return network['length'].sum()

def make_osrm_csv_file(edges, out_path):
    edges_flipped = [(v, u) for u, v in edges]

    df_ = pd.DataFrame(edges + edges_flipped, columns=["u", "v"])
    df_['speed'] = 20
    #df_['rate'] = 20
    df_.to_csv(out_path, index=False, header=None)

def calculate_path_gaps(condition, distances):
    # Modified based on https://stackoverflow.com/a/4495197/1218712

    distances = np.array(distances)

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return [sum(distances[start:stop]) for start, stop in idx]

#assert calculate_path_gaps(np.array([0, 0, 0, 1, 1, 1]) == 0, [10, 10, 20, 5, 10, 25]) == [40]
#assert calculate_path_gaps(np.array([1, 0, 0, 1, 1, 1]) == 0, [10, 10, 20, 5, 10, 25]) == [30]
#assert calculate_path_gaps(np.array([1, 1, 1, 1, 1, 1]) == 0, [10, 10, 20, 5, 10, 25]) == []
#assert calculate_path_gaps(np.array([1, 0, 0, 1, 0, 0]) == 0, [10, 10, 20, 5, 10, 25]) == [30, 35]
#assert calculate_path_gaps(np.array([0]) == 0, [100]) == [100]
#assert calculate_path_gaps(np.array([1, 1, 1, 1, 1, 0]) == 0, [10, 10, 20, 5, 10, 25]) == [25]
#assert calculate_path_gaps(np.array([0, 0, 0, 0, 0, 0]) == 0, [10, 10, 20, 5, 10, 25]) == [80]
#assert calculate_path_gaps(np.array([0, 0, 0, 0, 0, 1]) == 0, [10, 10, 20, 5, 10, 25]) == [55]
def calculate_and_classify_gaps(condition, distances):
    gaps = np.array(calculate_path_gaps(condition, distances))
    first_is_access = condition[0] == True
    last_is_access = condition[-1] == True

    access = None
    real_gaps = np.empty(0)
    egress = None

    if len(gaps) == 1:
        if first_is_access:
            access = gaps[0]
        elif last_is_access:
            egress = gaps[0]
        else:
            real_gaps = gaps
    elif len(gaps) >= 2:
        if first_is_access:
            access = gaps[0]
            gaps = np.delete(gaps, 0)
        if last_is_access:
            egress = gaps[-1]
            gaps = np.delete(gaps, -1)
        real_gaps = gaps
    return (access, real_gaps.tolist(), egress)

# assert calculate_and_classify_gaps(np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 1]) == 0, np.ones(10)) == (None, [2, 2], None)
# assert calculate_and_classify_gaps(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 0, np.ones(10)) == (None, [], None)
# assert calculate_and_classify_gaps(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == 0, np.ones(10)) == (10, [], None)
# assert calculate_and_classify_gaps(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) == 0, np.ones(10)) == (8, [], 1)
# assert calculate_and_classify_gaps(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1]) == 0, np.ones(10)) == (None, [2], None)
# assert calculate_and_classify_gaps(np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0]) == 0, np.ones(10)) == (3, [], 1)
# assert calculate_and_classify_gaps(np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0]) == 0, np.ones(10)) == (3, [2], 1)
# assert calculate_and_classify_gaps(np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 0]) == 0, np.ones(10)) == (3, [1, 1], 1)
# assert calculate_and_classify_gaps(np.array([0]) == 0, [10]) == (10, [], None)
# assert calculate_and_classify_gaps(np.array([1]) == 0, [10]) == (None, [], None)
# assert calculate_and_classify_gaps(np.array([0, 1]) == 0, [10, 20]) == (10, [], None)
# assert calculate_and_classify_gaps(np.array([1, 1]) == 0, [10, 20]) == (None, [], None)

def prepare_optimization(directory,
                         osm_xml_path,
                         lua_profile_path): # Path where to write the od_data dataframe

    # Create OSRM folder and copy the osm .xml file to it.
    directory = pathlib.Path(directory)
    osrm_folder = directory / 'osrm'
    osrm_folder.mkdir(parents=True, exist_ok=True)

    shutil.copy2(osm_xml_path, osrm_folder)
    shutil.copy2(lua_profile_path, osrm_folder)
    shutil.copytree(lua_profile_path.parent / 'lib', osrm_folder / 'lib', dirs_exist_ok=True)

    # Change the "csv_filter_path" variable in the lua profile.
    new_lua_profile_path = osrm_folder / lua_profile_path.name
    with open(new_lua_profile_path, 'r') as file :
        filedata = file.read()
    csv_filter_path = osrm_folder / "csv_filter.csv"
    filedata = filedata.replace('csv_filter_path', str(csv_filter_path))
    with open(new_lua_profile_path, 'w') as file:
        file.write(filedata)

def generate_individual(edges_df, max_length):
    individual = np.zeros(len(edges_df), dtype=int)
    network_length = 0

    while True:
        zeros_indices = np.where(individual == 0)[0]
        if zeros_indices.size == 0:
            # There are not more possible links to add
            break
            
        random_index = np.random.choice(zeros_indices)
        link_length = edges_df.iloc[random_index]['length']
        if network_length + link_length <= max_length:
            network_length += link_length
            individual[random_index] = 1
        else:
            break

    return list(individual)

def probabilistic_gene_crossover(ind1, ind2):
    ind1_fitness = ind1.fitness.values[0]
    ind2_fitness = ind2.fitness.values[0]

    # Probability of choosing from individual 1
    theta = ind2_fitness / (ind1_fitness + ind2_fitness)

    offspring = copy.deepcopy(ind1)
    del offspring.fitness.values

    for i in range(len(offspring)):
        if random.random() < theta:
            offspring[i] = ind1[i]
        else:
            offspring[i] = ind2[i]

    return offspring

def trips_based_crossover(ind1, ind2, edges_df, params):
    offspring = copy.deepcopy(ind1)
    for i in range(len(offspring)):
        offspring[i] = 0

    del offspring.fitness.values

    flux1 = ind1.trips
    flux2 = ind2.trips

    network_length = 0
    edge_lengths = edges_df['length'].values
            
    diffs = [abs(flux1[i] - flux2[i]) for i in range(len(offspring))]
    sorted_diffs = sorted(range(len(diffs)), key=lambda k: diffs[k], reverse=True)
    for idx in diffs:
        try:
            theta = flux1[idx] / (flux1[idx] + flux2[idx])
        except ZeroDivisionError:
            theta = 0.5

        chosen = ind1[idx] if random.random() < theta else ind2[idx]
        
        if network_length + edge_lengths[idx] <= params.MAX_CYCLING_NETWORK_LENGTH:
            network_length += edge_lengths[idx]
            offspring[idx] = chosen
        else:
            continue

    return offspring


from dataclasses import dataclass
@dataclass
class SimulationResult():
    fitness: int = np.inf
    total_duration: float = np.inf
    total_distance: float = np.inf
    total_trip_bike_distance: float = np.inf
    total_orig_dest_tot_disag_dist: float = np.inf
    unreachable_trips: int = np.inf
    network_length: float = np.inf
    trips: list = None

def calculate_fitness(individual,
                      edges_df,
                      working_directory, # The directory containing the 'osrm' folder as well as other relevant files (od_trips.csv, etc)
                      osrm_batch_router_path, # Full path to osrm-batch-router
                      od_df_exploded,
                      G,
                      optimization_params, # Optimization parameters
                      consider_constraints=True,
                      return_osrm_results=False,
                      n_retries=3):

    ret_val = None
    for i in range(n_retries):
        try:
            ret_val = _calculate_fitness(individual, edges_df, working_directory, osrm_batch_router_path, od_df_exploded, G, optimization_params, consider_constraints=consider_constraints, return_osrm_results=return_osrm_results)
            break
        except Exception as e:
            logging.exception("Encountered exception in calculate_fitness. Attempt #%d. Retrying after waiting a bit..." % i)
            time.sleep(1)
    else:
        logging.debug("Could not calculate fitness for individual with working directory %s. Returning insane fitness result." % working_directory)
        # Did not encouter "break", which means that we had n_retries exceptions in a row
        # We can return a SimulationResult which indicates failure.
        # FIXME: this should have a "failed" flag...
        ret_val = SimulationResult(fitness=9999999, total_duration=9999999, unreachable_trips=9999999, network_length=9999999, trips=None)
    return ret_val

# This is the fitness function for the genetic algorithm
# This code has to be fairly optimized because it will be executed thousands of times.
def _calculate_fitness(individual,
                      edges_df,
                      working_directory, # The directory containing the 'osrm' folder as well as other relevant files (od_trips.csv, etc)
                      osrm_batch_router_path, # Full path to osrm-batch-router
                      od_df_exploded,
                      G,
                      optimization_params, # Optimization parameters
                      consider_constraints,
                      return_osrm_results):

    network = edges_df[np.array(individual).astype(bool)]
    network_length = calculate_network_length(network)
    od_df_exploded = filter_exploded_od_df(network, G, od_df_exploded)

    working_directory = pathlib.Path(working_directory)
    osrm_folder = working_directory / 'osrm'

    # Remove everything that matches *.osrm.*
    for filename in osrm_folder.glob('*.osrm.*'):
        filename.unlink()

    # Create the csv filter file for exclusion of links from the osrm lua profile
    df_ = pd.DataFrame({'sequential_id': network.sequential_id})
    df_.to_csv(osrm_folder / "csv_filter.csv", index=False, header=None)        

    os.chdir(osrm_folder)
    osm_xml_file_path = list(osrm_folder.glob("*.xml"))[0] #/ pathlib.Path(osm_xml_path).name
    lua_profile_path = list(osrm_folder.glob("*.lua"))[0]
    for command in [["osrm-extract", "-p", lua_profile_path, osm_xml_file_path],
                    ["osrm-partition", osm_xml_file_path],
                    ["osrm-customize", osm_xml_file_path]]:

        result = subprocess.run(command, stdout=subprocess.PIPE)
        if result.returncode != 0:
            raise ValueError(f"OSRM command failed: {command[0]}: {result.stdout}")

    # Save the OD trips as a CSV to be used by the c++ OSRM program
    od_data_path = osrm_folder / 'od_trips.csv'
    od_df_for_writing = od_df_exploded[['lonorig', 'latorig', 'londest', 'latdest']]
    od_df_for_writing.to_csv(od_data_path, float_format='%.7f', index=True, header=False)

    result = subprocess.run([osrm_batch_router_path,
                            osm_xml_file_path,
                            od_data_path],
                            stdout=subprocess.PIPE)
    responses = ujson.loads(result.stdout.decode('utf-8'))

    od_df_exploded_copy = od_df_exploded.copy()
    
    # Any trips that are no longer in od_df_exploded are declared unreachable de facto
    unreachable_trips = len(set(range(0, od_df_exploded_copy.attrs['len_od_df'])) - set(od_df_exploded_copy.od_df_idx.unique()))

    for response in responses:
        if response['code'] == 'NoRoute':
            duration = np.nan
            distance = np.nan
            geometry = None
        else:
            route = response["routes"][0]
            distance = float(route['distance'])
            duration = float(route['duration'])
            geometry = response["routes"][0]["geometry"]
        row_id = int(response['id'])
        od_df_exploded_copy.at[row_id, 'trip_bike_distance'] = distance
        od_df_exploded_copy.at[row_id, 'trip_duration'] = duration
        od_df_exploded_copy.at[row_id, 'trip_geometry'] = geometry
    walking_speed = optimization_params.WALKING_SPEED / 3.6 # Convert km/h to m/s

    # The total duration and distance are the duration and distances on the network plus the duration and distances for the access and egress portions.
    od_df_exploded_copy['total_distance'] = od_df_exploded_copy['trip_bike_distance'] + od_df_exploded_copy['orig_dest_tot_disag_dist']
    od_df_exploded_copy['total_duration'] = od_df_exploded_copy['trip_duration'] + od_df_exploded_copy['orig_dest_tot_disag_dist'] / walking_speed

    min_idx = od_df_exploded_copy.groupby('od_df_idx')['total_duration'].idxmin(skipna=True) # For each od trip, take the one with the smallest total duration

    # The unreachable trips are the ones where the minimum duration is still np.nan
    unreachable_trips += min_idx.isna().sum()    
    # Now that we have counted the unreachable trips, we can drop them.
    min_idx = min_idx.dropna()

    extract = od_df_exploded_copy.loc[min_idx].set_index("od_df_idx")
    durations = extract['total_duration']
    total_distance = extract['total_distance'].fillna(0).sum()
    total_trip_bike_distance = extract['trip_bike_distance'].fillna(0).sum()
    total_orig_dest_tot_disag_dist = extract['orig_dest_tot_disag_dist'].fillna(0).sum()

    # Calculate the total duration of all trips ignore NaN's.
    total_duration = durations.fillna(0).sum()
    total_duration = total_duration / 3600 # seconds -> hours
    #network_length = network_length / 1000 # m -> km

    if consider_constraints and (network_length > optimization_params.MAX_CYCLING_NETWORK_LENGTH or network_length < 0.98*optimization_params.MAX_CYCLING_NETWORK_LENGTH):
        #print("adding penalty", network_length, MAX_CYCLING_NETWORK_LENGTH)
        penalty = optimization_params.CONSTRAINT_PENALTY
    else:
        penalty = 0

    fitness = optimization_params.VALUE_OF_TIME * total_duration \
            + optimization_params.UNREACHABLE_TRIP_COST * unreachable_trips \
            + penalty# + cost_per_km * network_length

    trips = None
    if hasattr(optimization_params, 'MATE') and optimization_params.MATE == "tripsBasedCrossover":
        edges_df['n_trips'] = 0

        all_edges_traversed = defaultdict(list)
        for r in responses:
            if r['code'] == 'NoRoute':
                continue
            # Ignore trips that are not the representative trip for the od pair.
            if r['id'] not in durations.index:
                continue
            nodes = r['routes'][0]['legs'][0]['annotation']['nodes']
            # Make sure we have integers
            nodes = [int(node) for node in nodes]
            for node_a, node_b in zip(nodes, nodes[1:]):
                    for node_order in [(node_a, node_b), (node_b, node_a)]:
                        #all_edges_traversed.append(index_order)
                        all_edges_traversed[node_order].append(r['id'])
        # map: apply __getitem__ for all edges in merged_edges
        # sum(x, []) flattens a shallow list of lists.
        # set: get unique trips
        # len: count them
        #trips = edges_df.apply(lambda row: len(set(sum(map(all_edges_traversed.__getitem__, row.merged_edges), []))), axis=1).to_list()
        trips = edges_df.apply(lambda row: len(set(all_edges_traversed.__getitem__((row.name[0], row.name[1])))), axis=1).to_list()
    result = SimulationResult(fitness=fitness,
                              total_duration=total_duration,
                              total_distance=total_distance,
                              total_trip_bike_distance=total_trip_bike_distance,
                              total_orig_dest_tot_disag_dist=total_orig_dest_tot_disag_dist,
                              unreachable_trips=unreachable_trips,
                              network_length=network_length,
                              trips=trips)
                              #feasible=True)
    return (result, responses, od_df_exploded_copy) if return_osrm_results else result
