import numpy as np 
import os
from PIL import Image

d = np.asarray(Image.open('../data/gmap_rail1/gmap_rail1_d.png'))

print(d.shape)
# describe the depth map (min, max, mean, etc)
print(f"min: {np.min(d)}, max: {np.max(d)}, mean: {np.mean(d)}")

print(d)