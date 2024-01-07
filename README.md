# Bicycle network design optimization

## Authors
1. David Murray, master's student at Polytechnique Montréal's [Mobility research chair](https://www.polymtl.ca/mobilite/)
2. Catherine Morency, PhD. Full professor and holder of the Mobility research chair.

## Questions
For any questions about the usage of this code, please open an issue in this repository

## Documentation
In addition to the README.md files in this repository, three other files are available as documentation:
1. [David Murray's master thesis](https://drive.google.com/file/d/1kTuoGaPsiHiOmf9ruYX4MlUdnu9qzl8k/view?usp=sharing) (in French only)
2. [TRB paper](https://drive.google.com/file/d/1VVYe4lEfTFNjvyqU_ZGSat-2CQGjj8sk/view?usp=sharing), presented at the 2024 TRB annual meetings.
3. [TRB poster](https://drive.google.com/file/d/1oawVAlYTGGMUJUdXUWnlu9ihhyC-V1K7/view?usp=sharing), presented at the 2024 TRB annual meetings.

## Code download and python environment setup
1. Install Python (follow instructions on https://www.python.org/downloads/)

Then, inside a terminal:
2. Clone the code from GitHub: `git clone https://github.com/davidmurray/cycling-network-design.git`
3. Change to the cloned directory: `cd cycling-network-design`
4. Create a virtual environment: `python -m venv .`
5. Activate the virtual environment: `source bin/activate` (note: command may differ on Windows)
6. Install the required packages: `pip install -r requirements.txt`

## Data preparation
The `code/data_preparation.py` Python script prepares the data necessary for the optimization process.

```bash
Parameters:

    --zone_polygon: Path to the GeoJSON file containing the area of the zone (required).
    --od_data: Path to the CSV file containing OD demand data (required).
    --output_directory: Path to the directory where the generated files will be saved (required).
    --local_crs: EPSG code for the local coordinate system (default: epsg:2950).
    --minimum_euclidean_distance: Minimum euclidean distance of trips to be retained (default: 1000 meters).
    --verbose: Enable verbose mode (default: False).
```

### Prerequisites
1. OD demand data in csv format. Your data can have many columns, but must at the very least have the `lonorig`, `londest`, `latorig` and `latdest` columns in WGS84 coordinates.
2. A polygon that defines the boundary of the area to optimize. Must be provided in GeoJSON format. An example is availble in the `examples/` folder.


### Usage

To run the script, use the following command in your terminal:

```bash

python code/data_preparation.py --zone_polygon /path/to/your/zone.geojson --od_data /path/to/your/od_data.csv --output_directory
```

Example output (with `verbose` turned on):
```bash
python code/data_preparation.py --zone_polygon test/boundary.geojson --od_data test/od_bike2018.csv --output_directory data/ --verbose
Starting data preparation for args test/boundary.geojson saving data to data/
  G = ox.graph_from_polygon(zone_polygon,
2024-01-02 12:10:07 Projected GeoDataFrame to 'EPSG:32618 / WGS 84 / UTM zone 18N'
2024-01-02 12:10:07 Projected GeoDataFrame to 'EPSG:4326 / WGS 84'
2024-01-02 12:10:07 Projected GeoDataFrame to 'EPSG:32618 / WGS 84 / UTM zone 18N'
2024-01-02 12:10:07 Projected GeoDataFrame to 'EPSG:4326 / WGS 84'
2024-01-02 12:10:07 Requesting data from API in 1 request(s)
2024-01-02 12:10:08 Retrieved response from cache file 'cache/0aa1f78f171879d4c08c90a41bba2a9c76189214.json'
2024-01-02 12:10:09 Retrieved all data from API in 1 request(s)
2024-01-02 12:10:09 Creating graph from 149,202 OSM nodes and 47,215 OSM ways...
2024-01-02 12:10:15 Created graph with 149,202 nodes and 359,054 edges
2024-01-02 12:10:17 Added length attributes to graph edges
2024-01-02 12:10:17 Identifying all nodes that lie outside the polygon...
2024-01-02 12:10:20 Created nodes GeoDataFrame from graph
2024-01-02 12:10:20 Built r-tree spatial index for 149,202 geometries
2024-01-02 12:10:20 Accelerating r-tree with 6 quadrats
2024-01-02 12:10:23 Identified 146,951 geometries inside polygon
2024-01-02 12:10:27 Removed 1,839 nodes outside polygon
2024-01-02 12:10:27 Truncated graph by polygon
2024-01-02 12:10:27 Identifying all nodes that lie outside the polygon...
2024-01-02 12:10:35 Created nodes GeoDataFrame from graph
2024-01-02 12:10:36 Built r-tree spatial index for 147,363 geometries
2024-01-02 12:10:36 Accelerating r-tree with 4 quadrats
2024-01-02 12:10:42 Identified 129,767 geometries inside polygon
2024-01-02 12:10:49 Removed 17,297 nodes outside polygon
2024-01-02 12:10:49 Truncated graph by polygon
2024-01-02 12:10:51 Counted undirected street segments incident on each node
2024-01-02 12:10:52 graph_from_polygon returned graph with 130,066 nodes and 313,758 edges
2024-01-02 12:10:56 Created nodes GeoDataFrame from graph
2024-01-02 12:10:57 Projected GeoDataFrame to 'EPSG:2950 / NAD83(CSRS) / MTM zone 8'
2024-01-02 12:11:01 Created edges GeoDataFrame from graph
2024-01-02 12:11:16 Created graph from node/edge GeoDataFrames
2024-01-02 12:11:16 Projected graph with 130066 nodes and 313758 edges
Dropped 27385 (81.2 %) trips. Remaining trips: 6356
2024-01-02 12:11:50 Created nodes GeoDataFrame from graph
2024-01-02 12:12:01 Created nodes GeoDataFrame from graph
2024-01-02 12:12:15 Counted undirected street segments incident on each node
2024-01-02 12:12:20 Created edges GeoDataFrame from graph
2024-01-02 12:12:30 Saved graph as GraphML file at PosixPath('data/G_consolidated.graphml')
2024-01-02 12:12:30 Begin topologically simplifying the graph...
2024-01-02 12:12:30 Identified 435 edge endpoints
2024-01-02 12:12:31 Simplified graph: 6,276 to 435 nodes, 13,440 to 1,758 edges
2024-01-02 12:12:32 Begin topologically simplifying the graph...
2024-01-02 12:12:32 Identified 372 edge endpoints
2024-01-02 12:12:32 Simplified graph: 435 to 372 nodes, 1334 to 1208 edges
2024-01-02 12:12:32 Created edges GeoDataFrame from graph
2024-01-02 12:12:33 Converted MultiDiGraph to undirected MultiGraph
2024-01-02 12:12:33 Saved graph as GraphML file at PosixPath('data/G_simplified.graphml')
2024-01-02 12:12:33 Created nodes GeoDataFrame from graph
2024-01-02 12:12:33 Created edges GeoDataFrame from graph
  osm_xml._save_graph_xml(
2024-01-02 12:12:37 Saved graph as .osm file at PosixPath('data/osrm_network.xml')
```
