import numpy as np
import networkx as nx
from keras.models import Sequential
from keras.layers.core import *
import time

# load shit up
file = np.load("profiles.npz")
users = file["arr_0"].astype("uint32")
profiles = file["arr_1"]
companies = file["arr_2"]
data = np.loadtxt("train.csv", dtype="uint32", delimiter=",", skiprows=1)
# sort the data
mask = data[:, 0] > data[:, 1]
data[mask] = data[mask][:, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]]
data = data[data[:, 0].argsort()]
edges = data[:, :2]
g = nx.Graph()
g.add_edges_from(edges)


# create a batch
batch_size = 10000
batch_half = 7500
# half is random
interactions = np.random.choice(users, size=(batch_size, 2))
labels = np.zeros((batch_size, 1))
# half is taken from the data
start = np.random.randint(0, high=(edges.size / 2 - batch_half))
interactions[batch_half:] = edges[start:start + 2500]
labels[batch_half:] = 1
# shuffle it
combined = np.hstack((interactions, labels))
np.random.shuffle(combined)
# and put back into two arrays for convenience
interactions = combined[:, :-1]
labels = combined[:, -1:]
# generate features for each interaction in the batch
batch = np.empty((batch_size, 5))
start = time.time()
for index, edge in enumerate(interactions[3:]):
    user1_friends = [i for i in g[edge[0]]]
    user2_friends = [i for i in g[edge[1]]]
    batch[index, 0] = len(user1_friends + user2_friends)
    mutual_friends = [i for i in user2_friends \
                        if i in user1_friends]
    batch[index, 1] = len(mutual_friends)
    try:
        path = nx.shortest_path(g, source=edge[0],
                                   target=edge[1])
        distance = len(path)
    except:
        distance = 100
    batch[index, 2] = distance
    user1_profile = profiles[int(edge[0])]
    user2_profile = profiles[int(edge[1])]
    batch[index, 3] = np.linalg.norm(user1_profile[1:].reshape(2, 3) - user2_profile[1:].reshape(2, 3)[::-1, :])
    batch[index, 4] = companies[min(user1_profile[0], user2_profile[0]),
                                       max(user1_profile[0], user2_profile[0])]
    m1 = min(edge[0], edge[1])
    m2 = max(edge[0], edge[1])
    filtered = edges[edges[:, 0] == m1]
    if filtered[filtered[:, 1] == m2].size != 0:
        labels[index] = 1
    if index % 1000 == 0:
        print(index, time.time() - start)

# create and train MLP
model = Sequential()
model.add(Dense(64, input_dim=5, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adagrad',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.load_weights('mlp.h5')
model.fit(batch, labels, nb_epoch=10, batch_size=32)
model.save_weights('mlp.h5', overwrite=True)


#model.predict(batch, batch_size=32)

#file = open("thing", "w")
#file.write(string)
#file.close()
