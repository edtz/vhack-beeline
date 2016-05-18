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
# prepare an mlp
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

predicted_i = 0
predicted_edges = np.empty((15000000, 3))
for user in users:
    # user's friends
    neighbours = set(i for i in g[user])
    # user's friends' friends ^ 5
    surrounding = set(i for i in nx.single_source_shortest_path_length(g, user, cutoff=2))
    # people who are possible friends
    surrounding -= neighbours
    # filter for possibilities we already evaluated
    surrounding = [i for i in surrounding if i > user]
    # roster of possible friends w/ possibility initialized to 0
    friendlist = np.zeros((len(surrounding), 2))
    friendlist[:, 0] = np.array(surrounding)
    # to avoid context switching, we feed nn a whole batch at once
    batch = np.empty((friendlist[:, 0].size, 5))
    # for each possible friend, we compute probability
    for index, possibility in enumerate(friendlist[:, 0].astype("uint32")):
        user1_friends = [i for i in g[user]]
        user2_friends = [i for i in g[possibility]]
        # friend counter
        batch[index, 0] = len(user1_friends + user2_friends)
        mutual_friends = [i for i in user2_friends \
                            if i in user1_friends]
        batch[index, 1] = len(mutual_friends)
        try:
            path = nx.shortest_path(g, source=user,
                                       target=possibility)
            distance = len(path)
        except:
            distance = 100
        batch[index, 2] = distance
        user1_profile = profiles[user]
        user2_profile = profiles[possibility]
        batch[index, 3] = np.linalg.norm(user1_profile[1:].reshape(2, 3) - user2_profile[1:].reshape(2, 3)[::-1, :])
        batch[index, 4] = companies[int(min(user1_profile[0], user2_profile[0])),
                                           int(max(user1_profile[0], user2_profile[0]))]
    if friendlist[:, 0].size != 0:
        friendlist[:, 1] = model.predict(batch, batch_size=32).reshape(1, -1)
        addition = friendlist[friendlist[:, 1].argsort()][::-1]
        addition = addition[addition[:, 1] == 1]
        slice_start = predicted_i
        slice_end = slice_start + addition[:, 0].size
        predicted_edges[slice_start:slice_end] = np.hstack((np.repeat(np.array([user]), addition[:, 0].size, axis=0).reshape(addition[:, 0].size, 1), addition))
        predicted_i = slice_end
        if user % 1000 == 0:
            print(user, predicted_i)
    if predicted_i > 14000000:
        predicted_edges[10000000:] = np.zeros_like(predicted_edges[10000000:])
        predicted_i = 10000000
np.savez("edges.npz", predicted_edges)
