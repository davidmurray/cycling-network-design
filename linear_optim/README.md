# Bicycle Network Design with Linear Programming

This Python script uses the PuLP optimization library for creating and solving a linear programming model for bicycle network design. The objective is to determine, within a given budget and considering different OD pairs and their transportation demands, the optimal subset of network edges to form a cycling network 

## Prerequisites

This code depends on the `pulp`, `networkx` and `matplotlib` python libraries. We recommend that you use the python environnement provided in this repo. See README.md file in the root of this repo for instructions on using this environment.

## Solver Configuration

Before running the script, ensure you have a linear programming solver installed. The script is configured to use IBM CPLEX on macOS. If you use a different solver, modify the `PATH_TO_SOLVER` variable accordingly.

## Demand Construction

Demand for commodities is created, specifying the origin and destination nodes for each commodity. Commodities represent different transportation needs within the network (od pairs).

# How to Use

1. Activate the python environnement
2. From this directory, run `python bicycle_network_design.py`

### Results Visualization

The results are visualized using Matplotlib. The selected edges are displayed in light green, unselected edges in light gray, and unreachable nodes in darker gray. The resulting graph is saved as `graph_result.svg` for visualization.
