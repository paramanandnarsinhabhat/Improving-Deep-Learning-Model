### 1. Importing Libraries and Data preprocessing
# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# set seed
seed = 42

# load csv file
data = pd.read_csv('data/Dataset/emergency_classification.csv')

print(data.head())