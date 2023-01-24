# Loading Required Libraries

from matplotlib import image
from matplotlib import pyplot
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
import glob

# Setting Homology Dimension


homology_dimensions = [1]
CP = CubicalPersistence(
    homology_dimensions=homology_dimensions,
    coeff=3,
    n_jobs=1
)
BC = BettiCurve()

# Reading images from file

file = "C://Users/16823/Desktop/Chest_X_Ray_Data/Pneumonia_Dataset/train/NORMAL/"

img_file = list(glob.glob1(file, "*.jpeg"))
img = []
for i in img_file:
    img.append(i)

# Feature Extraction

data = []
for i in img:
    image_path = file + i
    gray_h1 = Image.open(image_path).convert('L')
    im_gray_h1 = np.array(gray_h1)
    diagram_h1_0 = CP.fit_transform(np.array(im_gray_h1)[None, :, :])
    y_betti_curves_h1_0 = BC.fit_transform(diagram_h1_0)
    data.append(np.reshape(y_betti_curves_h1_0, 100))
df0 = pd.DataFrame(data)
df0["label"] = [0] * len(data)

df0.to_excel("df_train_n_Betti_1.xlsx")
