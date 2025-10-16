import csv

import numpy as np
from SSSOM import SemiSupervisedMiniSom

n_features = 46
n_labels = 23
csv_file = "artist_train_tanh_onehot.csv"
checkpoint_path = "./SSSOMweights/SemiSSOM_40_0.1_exponent_decay_euclidean_hexagonal_100.npy"
feature_labels = []
def load_dataset(csv_path):
    dataset = []
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        global feature_labels
        feature_labels = header[1:]
        for row in reader:
            features = row[1:]
            features = [float(x) for x in features]
            dataset.append(features)
    return np.array(dataset)

data = load_dataset(csv_file)

# for size in [25, 40]:
#     sigma = size / 2.0
#     for lr in [0.1, 0.5]:
#         for decay in ['asymptotic_decay', 'exponent_decay']:
#             for nf in ['euclidean', 'manhattan']:
#                 for topo in ['hexagonal', 'rectangular']:
#                     print(f"Training SOM with size={size}, lr={lr}, decay={decay}, distance={nf}, topology={topo}")
#                     som = SemiSupervisedMiniSom(x=size, y=size, feature_len=n_features, label_len=n_labels,
#                                                 sigma=sigma, learning_rate=lr, neighborhood_function='gaussian', activation_distance=nf, decay_function=decay, topology=topo)
#                     som.train_batch(data, epochs=10, checkpoint_interval=5000)
#                     som.plot_component_planes_hex(feature_names=feature_labels, hex_size=1, feature_indices=[i for i in range(69)][n_features:], save_dir=f"./component_planes/component_planes_hex_{size}_{lr}_{decay}_{nf}_{topo}")
                    # som.plot_label_map_hex(data=data, labels=feature_labels, hex_size=1, save_path=f"./label_map_hex_size{size}_lr{lr}_decay{decay}_dist{nf}_topo{topo}.png")
size = 40
sigma = size / 2.0
lr = 0.1
decay = 'exponent_decay'
nf = 'euclidean'
topo = 'hexagonal'
som = SemiSupervisedMiniSom(x=size, y=size, feature_len=n_features, label_len=n_labels, load_path=None,
                            sigma=sigma, learning_rate=lr, neighborhood_function='gaussian', activation_distance=nf, decay_function=decay, topology=topo)
som.train_batch(data, epochs=500, checkpoint_interval=13300)
som.plot_component_planes_hex(feature_names=feature_labels, hex_size=1, feature_indices=[i for i in range(69)][n_features:], save_dir=f"./component_planes/component_planes_hex_{size}_{lr}_{decay}_{nf}_{topo}_tanh")
som.plot_label_map_hex(data=data, labels=feature_labels, hex_size=1, save_path=f"./label_map_hex_size{size}_lr{lr}_decay{decay}_dist{nf}_topo{topo}_tanh.png")