import numpy as np
import networkx as nx

edges = np.load("edges.npz")["arr_0"][:, [0,1]].astype("uint32")
np.random.shuffle(edges)
edges = edges[:1000000]
g = nx.Graph()
g.add_edges_from(edges)
users = np.sort(g.nodes())
file = open("results.txt", "w")

for user in users:
    file.write(str(user))
    for edge in edges[edges[:, 0] == user]:
        file.write("," + str(edge[1]))
    for edge in edges[edges[:, 1] == user]:
        file.write("," + str(edge[0]))
    file.write("\n")
    if user % 1000 == 0:
        print(user)
file.close()
