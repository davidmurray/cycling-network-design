from pulp import *
import networkx as nx
import matplotlib.pyplot as plt

# Change this if needed! I used IBM CPLEX on macOS, but you can use other solvers.
PATH_TO_SOLVER = r'/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex'

print("Available solvers:", listSolvers(onlyAvailable=True))
##############################
# Network construction
##############################

G = nx.Graph()
H = nx.grid_graph(dim=[8, 10])

def index_to_letter_code(n):
   import string
   alphabet = string.ascii_uppercase
   base = 26
   i = n

   if i < base:
      return alphabet[i]
   else:
      return index_to_letter_code(i // base) + alphabet[i % base]
import string
node_names = list(string.ascii_uppercase) + ["A"+l for l in string.ascii_uppercase] + ["B"+l for l in string.ascii_uppercase] + ["C"+l for l in string.ascii_uppercase] + ["D"+l for l in string.ascii_uppercase]

H.remove_nodes_from([(1, 5), (1, 6), (5, 2), (8, 2), (6, 8), (2, 2), (6, 9), (9, 3), (9, 4), (3, 8), (3, 7), (4, 7), (4, 8), (3, 5), (3, 4), (4, 4), (4, 5), (8, 6), (5, 0), (8, 3)])

for (x, y), letter in zip(H.nodes(), node_names):
    G.add_node(letter, pos=(x, y))

for n1, n2 in H.edges:
    n1 = [n for n, data in G.nodes(data=True) if data['pos'] == n1][0]
    n2 = [n for n, data in G.nodes(data=True) if data['pos'] == n2][0]
    G.add_edge(n1, n2)

G.remove_edges_from([['A', 'B'],
                    ['K', 'G'],
                    ])

# Set all edge length and construction cost to one unit
for (i, j) in G.edges:
    G[i][j]['length'] = 1
    G[i][j]['cost'] = 1 # Construction cost

# We can change this code to have some edges be longer and have a larger cost.
# For example:
G['E']['F']['length'] = 5; G['E']['F']['cost'] = 5
G['I']['J']['length'] = 5; G['I']['J']['cost'] = 5

G = G.to_directed()

##############################
# Demand construction
##############################

# Create supply and demand
n_commodities = 11
commodities = list(range(1, n_commodities+1))
supply = {n:{k:0 for k in commodities} for n in G.nodes}
demand = {n:{k:0 for k in commodities} for n in G.nodes}

supply['A'][1] = 10; demand['P'][1] = 10
supply['P'][2] = 10; demand['A'][2] = 10
supply['D'][3] = 10; demand['Q'][3] = 10
supply['J'][4] = 10; demand['T'][4] = 10
supply['N'][5] = 10; demand['AJ'][5] = 10
supply['AN'][6] = 10; demand['U'][6] = 10
supply['BI'][7] = 10; demand['G'][7] = 10
supply['V'][8] = 10; demand['AM'][8] = 10
supply['BB'][9] = 10; demand['BJ'][9] = 10
supply['AT'][10] = 10; demand['AZ'][10] = 10

def supply_for_commodity(k):
    return max(values[k] for values in supply.values() if values[k] >= 0)

fig = plt.figure(figsize=(7, 7))
pos = {id_:(data['pos'][1], data['pos'][0]) for id_, data in G.nodes(data=True)}
nx.draw(G, pos=pos, node_color='lightgreen', with_labels=True, node_size=300)
fig.suptitle('Initial graph')
plt.savefig("initial_graph.svg", format="svg")

##############################
# Model construction
##############################

# Parameters
b = 70 # Total construction budget

# Create the linear programming model
model = LpProblem("Bicycle Network Design", LpMinimize)

# Decision variables
flow_vars = []
for i in G.nodes:
    for j in G.nodes:
        if (i, j) in G.edges:
            for k in commodities:
                flow_vars.append((i, j, k))

x = LpVariable.dicts("x", G.edges, cat=LpBinary)
y = pulp.LpVariable.dicts("y", flow_vars, lowBound=0, cat=LpInteger)
z = LpVariable.dicts("z", commodities, cat=LpBinary)
yz = pulp.LpVariable.dicts("yz", flow_vars, lowBound=0, cat=LpInteger)

penalty = 15 # Penalty for each unsatisfied trip

# Objective function
model += lpSum(G[i][j]['length'] * yz[i, j, k] for i, j, k in yz) + lpSum(penalty * (1 - z[k]) * supply_for_commodity(k) for k in commodities)

# Construction budget constraint
model += lpSum(data['cost'] * x[i, j] for i, j, data in G.edges(data=True)) <= b

# Flow conservation constraints
for i in G.nodes:
    for k in commodities:
        net_flow = (supply[i][k] - demand[i][k]) * z[k]
        model += pulp.lpSum(yz[i, j, k] for j in G.nodes if (i, j, k) in y) - \
                pulp.lpSum(yz[j, i, k] for j in G.nodes if (j, i, k) in y) == \
                net_flow, f"FlowConservation_({i},{j},{k})"

# Flow can only pass through constructed edges
for (i, j, k) in y:
    model += y[i, j, k] <= 10**5 * x[i, j]

# Constraints for yz, the product of y and z
for (i, j, k) in yz:
    model += yz[i, j, k] <= 10**5 * z[k]
    model += yz[i, j, k] <= y[i, j, k]
    model += yz[i, j, k] >= y[i, j, k] - (10**5)*(1 - z[k])
    model += yz[i, j, k] >= 0

# Binary variable constraints
for e in G.edges:
    model += x[e] in [0, 1]
for k in commodities:
    model += z[k] in [0, 1]

# Force two way edges
for (i, j) in G.edges:
    model += x[i, j] == x[j, i]

# You can write the model to an mps file to use it with other solvers or with the CPLEX gui software if you prefer.
#model.writeMPS("model.mps")

##############################
# Model solving
##############################

solver = CPLEX_CMD(path=PATH_TO_SOLVER, msg=True)
model.solve(solver)

# Print the status of the solution
print("Status:", LpStatus[model.status])

# Print the optimal objective value
print("Objective Value: ", value(model.objective))

# Print the constructed edges
print("Constructed Edges:")
for e in G.edges:
    print(f"{e} = {x[e].varValue}")

# Print the commodities: whether they are satisfied (z[k])
# and the transportation cost for this commodity.
print("Commodities:")
for k in commodities:
    cost = 0
    for i in G.nodes:
        for j in G.nodes:
            if (i, j, k) in y:
                #print(f"Flow from {i} to {j} --> {y[i, j, k].varValue} vs. {yz[i, j, k].varValue}")
                cost += G[i][j]['length'] * y[i, j, k].varValue
    print(f"{k} = {z[k].varValue}; cost = {cost}")

# Show selected edges and unselected edges and unreachable nodes
selected_edges    = [e for e in G.edges if x[e].varValue == 1.0]
unselected_edges  = [e for e in G.edges if x[e].varValue == 0.0]

fig = plt.figure(figsize=(7, 7))
nx.draw(G.edge_subgraph(selected_edges), 
        pos=pos,
        node_color='lightgreen',
        edge_color="lightgreen",
        with_labels=True,
        node_size=300)
nx.draw_networkx_edges(G.edge_subgraph(unselected_edges),
                       pos=pos,
                       edge_color="lightgray",
                       alpha=0.3,
                       node_size=300)
nx.draw_networkx_nodes(G.subgraph(G.nodes - G.edge_subgraph(selected_edges).nodes),
                       pos=pos,
                       node_color='lightgray',
                       alpha=0.3,
                       node_size=300)
nx.draw_networkx_labels(G.subgraph(G.nodes - G.edge_subgraph(selected_edges).nodes),
                       pos=pos,
                       alpha=0.3,
                       font_color="black")
plt.savefig("graph_result.svg", format="svg")
fig.suptitle('Results')

plt.show()
