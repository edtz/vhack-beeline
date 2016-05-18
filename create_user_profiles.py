# TODO convert to cython
# TODO refactor to one pass over data array
# jk aint nobody got time for that

import numpy as np
import networkx as nx

def profile_user(user, data):
	records = data[data[:, 0] == user]
	tx = records[:, [4,5,8]]
	rx = records[:, [6,7,9]]
	company = records[:1, 2]
	records = data[data[:, 1] == user]
	tx = np.concatenate([tx, records[:, [6,7,9]]])
	rx = np.concatenate([rx, records[:, [4,5,8]]])
	if company.size == 0:
		company = records[:1, 3]
	if tx.size != 0:
		tx_avg = np.average(tx, axis=0)
	else:
		tx_avg = np.array([0., 0., 0.])
	if rx.size != 0:
		rx_avg = np.average(rx, axis=0)
	else:
		rx_avg = np.array([0., 0., 0.])
	return np.array([company[0], tx_avg[0], tx_avg[1], tx_avg[2], rx_avg[0], rx_avg[1], rx_avg[2]])

def profile_companies(data):
	relation_matrix = np.zeros((5, 5), dtype="uint32")
	for i in data:
		if i[0]<i[1]:
			relation_matrix[i[0], i[1]] += 1
		else:
			relation_matrix[i[1], i[0]] += 1
	ratio_matrix = relation_matrix / data.size
	return ratio_matrix


data = np.loadtxt("train.csv", dtype="uint32", delimiter=",", skiprows=1)
edges = data[:, :2]
g = nx.Graph()
g.add_edges_from(edges)
total_users = g.number_of_nodes()
users = np.sort(g.nodes())
companies_matrix = profile_companies(data[:, 2:4])
profiles = np.zeros((1216100, 7))

for user in users:
	profiles[user] = profile_user(user, data)
	if user % 1000 == 0:
		print(user)

np.savez("profiles.npz", users, profiles, companies_matrix)
