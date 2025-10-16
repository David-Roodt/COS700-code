import csv
import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from minisom import MiniSom
from tqdm import tqdm


def _hex_coords(m, n, size=1):
    coords = []
    for r in range(m):
        for c in range(n):
            x = c * np.sqrt(3) * size
            y = r * 1.5 * size
            if c % 2 == 1:
                y += 0.75 * size
            coords.append((x, y))
    return np.array(coords).reshape(m, n, 2)

def exponent_decay(val0, iter, max_iter):
    if(val0 <= 1):
        val_final = 0.000001
    else:
        val_final = 1
    tau = max_iter / np.log(val0 / val_final)
    return val0 * np.exp(-iter / tau)

class SemiSupervisedMiniSom(MiniSom):
    def __init__(self, x, y, feature_len, label_len=0, sigma=1.0, learning_rate=0.5, decay_function='asymptotic_decay', neighborhood_function='gaussian', activation_distance='euclidean', topology='hexagonal', random_seed=None, load_path=None):
        matplotlib.use('agg')  
        self.label_len = label_len
        self.feature_len = feature_len
        self.decay_name = decay_function
        self.activation_name = activation_distance
        if(decay_function == 'exponent_decay'):
            decay_function = exponent_decay

        if load_path and os.path.exists(load_path):
            print(f"Loading SOM weights from {load_path}")
            self._weights = np.load(load_path)
            all_labels = self._weights[:, :, self.feature_len:]
            self.label_mean = all_labels.mean(axis=(0, 1))
            self.label_std = all_labels.std(axis=(0, 1))
        super().__init__(x=x, y=y, input_len=feature_len + label_len, sigma=sigma, learning_rate=learning_rate, neighborhood_function=neighborhood_function, decay_function=decay_function, random_seed=random_seed, topology=topology, activation_distance=activation_distance)

    def winner(self, x):
        feats = x[:self.feature_len] if self.label_len > 0 else x
        min_dist = np.inf
        bmu_idx = (0, 0)
        for i in range(self._weights.shape[0]):
            for j in range(self._weights.shape[1]):
                w_feats = self._weights[i, j, :self.feature_len] if self.label_len > 0 else self._weights[i, j, :]
                dist = np.linalg.norm(feats - w_feats)
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        return bmu_idx

    def should_stop(self, qe_history, window=5, tol=0.001):
        if len(qe_history) < window + 1:
            return False
        recent = qe_history[-(window+1):]
        thes = max(recent) * tol
        deltas = [abs((recent[i+1] - recent[i]) / recent[i]) for i in range(window)]
        return all(delta < thes for delta in deltas)

    def train_batch(self, data, epochs, load_path=None, checkpoint_interval=1000):
        start_iter = 0
        epochs_recorded = []  
        q_errors = []
        if load_path and os.path.exists(load_path):
            print(f"Loading checkpoint from {load_path}")
            self._weights = np.load(load_path)
            start_iter = int(os.path.basename(load_path).split('_')[-1].split('.')[0])
            print(f"Resuming from epoch {start_iter}")

            qe = self.quantization_error(data)
            q_errors.append(qe)
            epochs_recorded.append(start_iter)
            print('\n quantization error:', qe)
            print('\n topographic error:', self.topographic_error(data))

        print(f"Before running training for {epochs} epochs")

        # qe = self.quantization_error(data)
        # q_errors.append(qe)
        # epochs_recorded.append(start_iter)
        # print('\n quantization error:', qe)
        # print('\n topographic error:', self.topographic_error(data))
        current_iter = start_iter
        def get_decay_rate(iteration_index, data_len):
            return int(iteration_index / data_len)
        for epoch in range(epochs):
            data = np.random.permutation(data)
            current_iter = start_iter + epoch + 1
            with open("ProgressBar-SOM.txt", "w", encoding="utf-8") as log_file:
                for iteration in tqdm(range(len(data)), desc=f"Running Epoch {epoch}/{epochs}", file=log_file):
                    self.update(data[iteration], self.winner(data[iteration]), get_decay_rate(iteration + epoch * len(data), epochs * len(data)), epochs * len(data))
                    if checkpoint_interval != -1 and ((epoch * len(data)) + iteration + 1) % checkpoint_interval == 0:
                        checkpoint_file = f"./SSSOMweights/SemiSSOM_{len(self._neigx)}_{self._learning_rate}_{self.decay_name}_{self.activation_name}_{self.topology}_{current_iter:03d}_tanh.npy"
                        np.save(checkpoint_file, self._weights)
                        print(f"Checkpoint saved at iteration {iteration + 1} as {checkpoint_file}")
                        qe = self.quantization_error(data)
                        q_errors.append(qe)
                        epochs_recorded.append(current_iter)
                        print('quantization error:', qe)
                        print('topographic error:', self.topographic_error(data))
            qe = self.quantization_error(data)
            q_errors.append(qe)
            epochs_recorded.append(current_iter)
            if self.should_stop(q_errors):
                print(f"Stopping early at epoch {epoch}, QE stabilized.")
                break

        checkpoint_file = f"./SSSOMweights/SemiSSOM_{len(self._neigx)}_{self._learning_rate}_{self.decay_name}_{self.activation_name}_{self.topology}_{current_iter:02d}_tanh.npy"
        np.save(checkpoint_file, self._weights)
        all_labels = self._weights[:, :, self.feature_len:]
        self.label_mean = all_labels.mean(axis=(0, 1))
        self.label_std = all_labels.std(axis=(0, 1))
        print(f"Checkpoint saved after {current_iter} epochs to {checkpoint_file}")
        qe = self.quantization_error(data)
        q_errors.append(qe)
        epochs_recorded.append(current_iter)

        plt.figure()
        plt.plot(epochs_recorded, q_errors, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Quantization Error")
        plt.title("Quantization Error over Epochs")
        plt.savefig(f"./QE/{len(self._neigx)}_{self._learning_rate}_{self.decay_name}_{self.activation_name}_{self.topology}_{current_iter}_tanh Quantization Error over Epochs {start_iter}-{current_iter}.png")
        plt.close()
        plt.figure()
        plt.plot(epochs_recorded[-10:], q_errors[-10:], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Quantization Error")
        plt.title("Quantization Error over Epochs")
        plt.savefig(f"./QE/{len(self._neigx)}_{self._learning_rate}_{self.decay_name}_{self.activation_name}_{self.topology}_{current_iter}_tanh Quantization Error over last 10 Epochs.png")
        plt.close()
        print('quantization error:', qe)
        print('topographic error:', self.topographic_error(data))

        final_qe = q_errors[-1]
        best_qe = min(q_errors)
        best_epoch = epochs_recorded[q_errors.index(best_qe)]
        
        params = {
            "size": len(self._neigx),
            "learning_rate": self._learning_rate,
            "decay_rate": self.decay_name,
            "activation_function": self.activation_name,
            "topology": self.topology,
            "final_sigma": self._sigma,   # or whatever you use
        }

        # Prepare row for CSV
        row = {
            **params,
            "final_qe": final_qe,
            "best_qe": best_qe,
            "best_epoch": best_epoch,
        }
        csv_path = "./SOMResults.csv"
        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def plot_component_planes_hex(self, feature_names=None, hex_size=1, feature_indices=None, save_dir="./component_planes_hex"):
        m, n, dim = self._weights.shape
        coords = _hex_coords(m, n, size=hex_size)

        os.makedirs(save_dir, exist_ok=True)

        for d in range(dim):
            if feature_indices is not None and d not in feature_indices:
                continue

            plt.figure(figsize=(8, 6))
            values = self._weights[:, :, d]

            vmin, vmax = values.min(), values.max()
            for r in range(m):
                for c in range(n):
                    x, y = coords[r, c]
                    color_val = (values[r, c] - vmin) / (vmax - vmin + 1e-9)
                    hexagon = RegularPolygon(
                        (x, y), numVertices=6, radius=hex_size, 
                        facecolor=plt.cm.viridis(color_val),
                        edgecolor="k"
                    )
                    plt.gca().add_patch(hexagon)

            title = f"Component {d}" if feature_names is None else feature_names[d]
            plt.title(title)
            plt.axis("equal")
            plt.axis("off")
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax))
            sm.set_array([])  # required dummy data

            fig = plt.gcf()
            ax = plt.gca()
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label(title)
            save_path = os.path.join(save_dir, f"{title}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Saved {save_path}")

    def plot_label_map_hex(self, data, labels, hex_size=1, save_path="./label_map_hex.png"):
        m, n, _ = self._weights.shape
        coords = _hex_coords(m, n, size=hex_size)

        features = [x[:self.feature_len] for x in data]
        label_vectors = [x[-self.label_len:] for x in data]
        
        bmus_features = [self.winner(x) for x in features]
        
        padded_labels = [
            np.concatenate([np.zeros(self.feature_len), lbl])
            for lbl in label_vectors
        ]
        bmus_labels = [self.winner(x) for x in padded_labels]

        # Collect labels for each neuron
        feature_map = {(r, c): [] for r in range(m) for c in range(n)}
        label_map = {(r, c): [] for r in range(m) for c in range(n)}
        for bmu, lbl in zip(bmus_features, labels):
            feature_map[bmu].append(lbl)

        for bmu, lbl in zip(bmus_labels, labels):
            label_map[bmu].append(lbl)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        for r in range(m):
            for c in range(n):
                x, y = coords[r, c]
                if feature_map[(r, c)]:
                    majority_label = Counter(feature_map[(r, c)]).most_common(1)[0][0]
                    text = str(majority_label)
                    facecolor = "lightblue"
                else:
                    text = ""
                    facecolor = "white"

                hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_size,
                                        facecolor=facecolor, edgecolor="k")
                axes[0].add_patch(hexagon)
                axes[0].text(x, y, text, ha="center", va="center", fontsize=8)

        axes[0].set_title("Label Map")
        axes[0].axis("equal")
        axes[0].axis("off")

        for r in range(m):
            for c in range(n):
                x, y = coords[r, c]
                if label_map[(r, c)]:
                    majority_label = Counter(label_map[(r, c)]).most_common(1)[0][0]
                    text = str(majority_label)
                    facecolor = "lightgreen"
                else:
                    text = ""
                    facecolor = "white"
                hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_size,
                                        facecolor=facecolor, edgecolor="k")
                axes[1].add_patch(hexagon)
                axes[1].text(x, y, text, ha="center", va="center", fontsize=8)

        axes[1].set_title("Label Map")
        axes[1].axis("equal")
        axes[1].axis("off")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved feature and label maps to {save_path}")

    def predict_labels_std(self, x, label_names):
        bmu = self.winner(x)
        full_label_vector = self._weights[bmu[0], bmu[1], self.feature_len:]
        significant_labels = [(name, val) for name, val, mean, std in zip(label_names, full_label_vector, self.label_mean, self.label_std)
                            if val > mean + std]
        significant_labels.sort(key=lambda x: x[1], reverse=True)
        
        pred_binary = (full_label_vector > self.label_mean + self.label_std).astype(int)

        return bmu, full_label_vector, significant_labels, pred_binary