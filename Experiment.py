import argparse
import csv
import random
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from dict import get_random_artist, get_random_artwork, load_artist_dict
from FeatureExtraction import Pipeline, pipeline_options
from GenAI import GenAIModel
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from SSSOM import SemiSupervisedMiniSom
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run SOM test evaluation")
parser.add_argument("--num_tests", type=int, default=50, help="Number of tests to run")
parser.add_argument("--debug", action="store_true", help="Enable debug prints")
parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation")

args = parser.parse_args()
Debug = args.debug
num_tests = args.num_tests
if args.seed is not None:
    random.seed(args.seed)

# Setup Models
model_dir = Path("./fine_tuned_sd2_1")
checkpoint_path = "./SSSOMweights/SemiSSOM_40_0.1_exponent_decay_euclidean_hexagonal_500_tanh.npy"
artist_dict = load_artist_dict("artist_dict.json")
RandomForest_path = "./RandomForest/artist_train_tanh.joblib"
            
rf = joblib.load(RandomForest_path)
gen_ai = GenAIModel(model_dir)
pipeline = Pipeline(pipeline_options, k=20, c=0.01, k_cluster=5, number_of_regions_for_comp=3, number_of_regions_for_color=5)
som = SemiSupervisedMiniSom(x=40, y=40, feature_len=46, label_len=23,
                            sigma=20.0, learning_rate=0.1, neighborhood_function='gaussian', activation_distance='euclidean', 
                            topology='hexagonal', random_seed=42, load_path=checkpoint_path)

output_csv = Path("SOM_Evaluation_1Artist.csv")
header = ["Prompt", "GroundTruth", "BMU", "Accuracy", "Precision", "Recall", "F1"]

if not output_csv.exists():
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


with open("ProgressBar-FullTests.txt", "w", encoding="utf-8") as log_file:
    for test_idx in tqdm(range(num_tests), file=log_file):
        # Artist = get_random_artist(artist_dict)
        Artwork_Artist, Artwork = get_random_artwork(artist_dict)
        prompt = f"{Artwork} by {Artwork_Artist}"
        if Debug:
            # print(f"Selected Artist: {Artist}")
            print(f"Selected Artwork: {Artwork}")
            print(f"Generated Prompt: {prompt}")
            print(f"Selected Artwork Artist: {Artwork_Artist}")

        _ = gen_ai.generate_images(prompt, num_images=1)
        full_image_path = Path(f"./GeneratedImages/{prompt}.png")
        result_features = pipeline.run(full_image_path)
        if Debug:
            print(f"Result_features: {result_features}\n")
            print(f"Result_features Length: {len(result_features)}\n")
        # numeric_features = [f"{Decimal(feature):.10f}" for feature in result_features]
        # if Debug:
        #     print(f"Numeric Features: {numeric_features}\n")
        #     print(f"Numeric Features Length: {len(numeric_features)}\n")

        label_names = list(artist_dict.keys())
        bmu, full_label_vector, significant_labels, pred_binary = som.predict_labels_std(result_features, label_names)
        if Debug:
            print(f"Best Matching Unit: {bmu}\n")
            print(f"Full Label Vector: {full_label_vector}\n")
            print(f"Prediction Binary: {pred_binary}\n")
        print(f"Significant Labels: {significant_labels}\n")
        # print(f"The contributing artists for the generated image were: {Artist} and {Artwork_Artist}\n")
        print(f"The contributing artists for the generated image were: {Artwork_Artist}\n")

        ground_truth = [0] * len(label_names)
        # if Artist in label_names:
        #     ground_truth[label_names.index(Artist)] = 1
        if Artwork_Artist in label_names:
            ground_truth[label_names.index(Artwork_Artist)] = 1

        accuracy = accuracy_score(ground_truth, pred_binary)
        precision = precision_score(ground_truth, pred_binary, zero_division=0)
        recall = recall_score(ground_truth, pred_binary, zero_division=0)
        f1 = f1_score(ground_truth, pred_binary, zero_division=0)
        if Debug:
            print(f"Ground Truth: {ground_truth}\n")
            print(f"Predicted Binary: {pred_binary}\n")
            print(f"Accuracy: {accuracy}\n")
            print(f"Precision: {precision}\n")
            print(f"Recall: {recall}\n")
            print(f"F1 Score: {f1}\n")
        rf_prediction = rf.predict([result_features])
        rf_probs = rf.predict_proba([result_features])
        rf_threshold = 0.05
        tf_binary = (rf_probs[0] > rf_threshold).astype(int)
        
        rf_accuracy = accuracy_score(ground_truth, tf_binary)
        rf_precision = precision_score(ground_truth, tf_binary, zero_division=0)
        rf_recall = recall_score(ground_truth, tf_binary, zero_division=0)
        rf_f1 = f1_score(ground_truth, tf_binary, zero_division=0)
        if Debug:
            print(f"Random Forest Prediction: {rf_prediction}\n")
            print(f"Random Forest Probabilities: {rf_probs}\n")
            print(f"Random Forest Binary (threshold={rf_threshold}): {tf_binary}\n")
            print(f"Random Forest Accuracy: {rf_accuracy}\n")
            print(f"Random Forest Precision: {rf_precision}\n")
            print(f"Random Forest Recall: {rf_recall}\n")
            print(f"Random Forest F1 Score: {rf_f1}\n")
            for i, sample_probs in enumerate(rf_probs):
                top_indices = np.argsort(sample_probs)[::-1]
                for rank, idx in enumerate(top_indices):
                    print(f"  Rank {rank+1}: f{label_names[idx]} ({sample_probs[idx]*100:.2f}%)")
                    if(label_names[idx] == Artwork_Artist):
                        rf_rank = rank + 1
                        break;
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # writer.writerow([prompt, f"{Artist}/{Artwork_Artist}:{Artwork}", bmu, accuracy, precision, recall, f1])
            writer.writerow([prompt, bmu, accuracy, precision, recall, f1, rf_rank, rf_accuracy, rf_precision, rf_recall, rf_f1])