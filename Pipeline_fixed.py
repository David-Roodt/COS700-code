import os
import random
import sys
import time
from decimal import Decimal, getcontext
from functools import partial

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from tqdm import tqdm


def average_L(image):
  return np.mean(image[:, :, 0])

def average_A(image):
  return (np.mean(image[:, :, 1]) + 1) / 2

def average_B(image):
  return (np.mean(image[:, :, 2]) + 1) / 2

def oklab_hue_angle(image):
  return np.mean((np.arctan2(image[:, :, 2], image[:, :, 1]) + np.pi) / (2*np.pi))

def oklab_hue_histogram(image, k):
  a = image[:, :, 1]
  b = image[:, :, 2]
  hue_angle = np.arctan2(b, a)
  hue_deg = (np.degrees(hue_angle) + 360) % 360
  hist, _ = np.histogram(hue_deg, bins=k, range=(0, 360))
  hist = hist / np.sum(hist)
  return hist.tolist()

def horizontal_coords(image, props, number_of_regions_for_comp):
  topreg = sorted(props, key=lambda p: p.area, reverse=True)[:number_of_regions_for_comp]
  h, w, _ = image.shape
  res = []
  for region in topreg:
    coords = region.coords

    norm_coords = np.array([(y / h, x / w) for y, x in coords])
    yk = norm_coords[:, 0]

    res.append(np.mean(yk))
  res.extend([0.0] * (number_of_regions_for_comp - len(res)))
  return res

def vertical_coords(image, props, number_of_regions_for_comp):
  topreg = sorted(props, key=lambda p: p.area, reverse=True)[:number_of_regions_for_comp]
  h, w, _ = image.shape
  res = []
  for region in topreg:
    coords = region.coords

    norm_coords = np.array([(y / h, x / w) for y, x in coords])
    xk = norm_coords[:, 1]

    res.append(np.mean(xk))
  res.extend([0.0] * (number_of_regions_for_comp - len(res)))
  return res

def mass_variance(image, props, number_of_regions_for_comp):
  topreg = sorted(props, key=lambda p: p.area, reverse=True)[:number_of_regions_for_comp]
  h, w, _ = image.shape
  res = []
  for region in topreg:
    centroid = region.centroid
    x_bar, y_bar = centroid[1] / w, centroid[0] / h
    coords = region.coords

    norm_coords = np.array([(y / h, x / w) for y, x in coords])
    xk, yk = norm_coords[:, 1], norm_coords[:, 0]

    res.append(np.mean((xk - x_bar) ** 2 + (yk - y_bar) ** 2))
  res.extend([0.0] * (number_of_regions_for_comp - len(res)))
  return res

def mass_skewness(image, props, number_of_regions_for_comp):
  topreg = sorted(props, key=lambda p: p.area, reverse=True)[:number_of_regions_for_comp]
  h, w, _ = image.shape
  res = []
  for region in topreg:
    centroid = region.centroid
    x_bar, y_bar = centroid[1] / w, centroid[0] / h
    coords = region.coords

    norm_coords = np.array([(y / h, x / w) for y, x in coords])
    xk, yk = norm_coords[:, 1], norm_coords[:, 0]

    val = np.mean((xk - x_bar) ** 3 + (yk - y_bar) ** 3)
    res.append((np.tanh(val * 3) + 1) / 2)
  res.extend([0.0] * (number_of_regions_for_comp - len(res)))
  return res

def average_hue_angle_per_region(image, props, number_of_regions_for_color):
  topreg = sorted(props, key=lambda p: p.area, reverse=True)[:number_of_regions_for_color]
  res = []
  for region in topreg:
    coords = region.coords
    a_vals = np.array([image[y, x, 1] for y, x in coords])
    b_vals = np.array([image[y, x, 2] for y, x in coords])
    hue_angle = np.arctan2(b_vals, a_vals)
    hue_deg = (np.degrees(hue_angle) + 360) % 360
    avg_hue_deg = np.mean(hue_deg)
    res.append(avg_hue_deg / 360.0)
  res.extend([0.0] * (number_of_regions_for_color - len(res)))
  return res

def average_lightness_per_region(image, props, number_of_regions_for_color):
  topreg = sorted(props, key=lambda p: p.area, reverse=True)[:number_of_regions_for_color]
  res = []
  for region in topreg:
    coords = region.coords
    L_vals = np.array([image[y, x, 0] for y, x in coords])
    res.append(np.mean(L_vals))
  res.extend([0.0] * (number_of_regions_for_color - len(res)))
  return res

class Pipeline:
  def __init__(self, options, k=None, c=None, k_cluster=None, number_of_regions_for_comp=None, number_of_regions_for_color=None):
    self.options = []
    self.k_cluster = k_cluster
    self.props = None
    self.coloured_image = None
    for opt in options:
      if opt == oklab_hue_histogram:
        opt = partial(opt, k=k)
        opt._returns_list = True
        opt._needs_segmentation = False
      elif opt in [horizontal_coords, vertical_coords, mass_variance, mass_skewness]:
        opt = partial(opt, number_of_regions_for_comp=number_of_regions_for_comp)
        opt._returns_list = True
        opt._needs_segmentation = True
      elif opt in [average_hue_angle_per_region, average_lightness_per_region]:
        opt = partial(opt, number_of_regions_for_color=number_of_regions_for_color)
        opt._returns_list = True
        opt._needs_segmentation = True
      else:
        opt._returns_list = False
        opt._needs_segmentation = False
      func = opt.func if isinstance(opt, partial) else opt
      if func in [average_L, average_A, average_B, oklab_hue_angle, oklab_hue_histogram, 
              average_hue_angle_per_region, average_lightness_per_region]:
        opt._needs_coloured_image = True
      else:
        opt._needs_coloured_image = False
      self.options.append(opt)

  def run(self, image_path):
    image = cv2.imread(image_path)
    res = []
    if(any(opt._needs_coloured_image for opt in self.options)):
      self.coloured_image = self.bgr_to_oklab(image)
    if(any(opt._needs_segmentation for opt in self.options)):
      self.segment(image, self.k_cluster)
    for option in self.options:
      if option._needs_segmentation:
        if option._needs_coloured_image:
          temp = option(self.coloured_image, self.props)
        else:
          temp = option(image, self.props)
      else:
        if option._needs_coloured_image:
          temp = option(self.coloured_image)
        else:
          temp = option(image)
      if option._returns_list:
        res.extend(temp)
      else:
        res.append(temp)
        
    for i in range(len(res)):
      val_dec = Decimal(float(res[i]))
      res[i] = f"{val_dec:.10f}"
    return ",".join(res)

  def segment(self, image, k):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, _ = image.shape
    reshaped = image_hsv.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(reshaped)
    labels = kmeans.labels_.reshape((h, w))
    output = np.zeros_like(labels, dtype=np.uint8)
    for label_id in range(k):
        mask = (labels == label_id).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        output[mask == 1] = label_id + 1
    unique, counts = np.unique(output[output > 0], return_counts=True)
    sorted_regions = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
    top_regions = [region_id for region_id, _ in sorted_regions[:len(sorted_regions) // 2]]
    markers = np.zeros_like(output)
    for region_id in top_regions:
        markers[output == region_id] = region_id
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distance = ndi.distance_transform_edt(gray)
    segmented = watershed(-distance, markers, mask=gray > 0)
    self.props = regionprops(label(segmented))

  def bgr_to_oklab(self, image_bgr):
    if image_bgr.shape[-1] > 3:
      image_bgr = image_bgr[:, :, :3]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    a = 0.055
    linear_rgb =  np.where(img_rgb <= 0.04045,
                    img_rgb / 12.92,
                    ((img_rgb + a) / (1 + a)) ** 2.4)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                  [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = linear_rgb @ M1.T
    lms_cbrt = np.cbrt(lms)
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                  [1.9779984951, -2.4285922050, 0.4505937099],
                  [0.0259040371, 0.7827717662, -0.8086757660]])
    oklab = lms_cbrt @ M2.T
    return oklab


BATCH_SIZE = 51
data_path = "./wikiart/"
file_paths = {
    'Test': "./Book1.csv"
}
pipeline_options = [average_L, average_A, average_B, oklab_hue_angle, oklab_hue_histogram, 
                    horizontal_coords, vertical_coords, mass_variance, mass_skewness, 
                    average_hue_angle_per_region, average_lightness_per_region]
flagged = False
done = False
process = "./rejects.csv"
pipelined = "./Book1.csv"
pipeline = Pipeline(pipeline_options, k=20, c=0.01, k_cluster=5, number_of_regions_for_comp=3, number_of_regions_for_color=5)
temp_file_path = "batch.tmp"
j = 1
for i in range(5):
  if flagged:
    print("Flagged an error, stopping.")
    break
  with open(process, 'r', encoding='utf-8') as original_file:
      lines = original_file.readlines()
  if not lines:
      print("All lines processed.")
      flagged = True
  batch = lines[:BATCH_SIZE]
  remaining = lines[BATCH_SIZE:]
  with open(process, 'w', encoding='utf-8') as original_file:
    original_file.write("".join(remaining))
  with open("./errors.txt", 'w', encoding='utf-8') as original_file:
    original_file.write("")
  error_lines = []
  done_lines = 0
  total_lines = 0
  with open(process, 'r', encoding='utf-8') as f:
    for l in f:
      if len(l.strip().split(",")) != 2:
          done_lines += 1
      total_lines += 1

  print(f"Processing {len(batch)} lines from file: {process} (Total lines: {total_lines}, Done lines: {done_lines})")
  with open("errors.txt", "w", encoding="utf-8") as log_file:
    for line in tqdm(batch, desc="Processing 1 batch", file=log_file):
      try:
        original_line = line.strip()
        if not original_line or original_line.lower().startswith("(path"):
          continue
        columns = original_line.split(",")
        if len(columns) != 2:
          continue
        image_path = columns[0].strip()
        image_class = columns[1].strip()
        full_image_path = os.path.join(data_path, image_path)
        if not os.path.exists(full_image_path):
          print(f"Missing: {image_path}")
          error_lines.append(original_line)
          continue
        result_features = pipeline.run(full_image_path)
        if(len(result_features.split(",")) != 46):
          print(f"Error with line: {result_features}\nLen: {len(result_features.split(','))}\n")
          error_lines.append(original_line)
        else:
          new_line = original_line + "," + result_features
          with open(pipelined, 'a', encoding='utf-8') as original_file:
            original_file.write(new_line + "\n")
      except Exception as e:
        print(f"Error processing line: {line.strip()}. Error: {e}")
        error_lines.append(original_line)
        flagged = True

    with open(process, 'a', encoding='utf-8') as original_file:
      original_file.write("\n".join(error_lines) + "\n")
    print(f"One batch of {len(batch)} lines processed. {len(error_lines)} lines were incorrect.")
