import pandas as pd
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # New import for Support Vector Classifier (SVM)
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
from google.colab import files
import matplotlib.pyplot as plt
import seaborn as sns

# --- Google Colab Specific File Handling ---
print("Please upload your 'ADNIMERGE.csv' file:")
uploaded = files.upload()

file_name = 'ADNIMERGE.csv'
if file_name in uploaded:
    data = pd.read_csv(io.BytesIO(uploaded[file_name]))
    print(f"'{file_name}' uploaded and loaded successfully into a Pandas DataFrame!")
else:
    print(f"Error: '{file_name}' not found in uploaded files. Please ensure the file is named correctly and was uploaded.")
    raise FileNotFoundError(f"'{file_name}' not uploaded.")

# --- Configuration ---
TARGET_COLUMN = 'DX'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MISSING_VALUE_THRESHOLD = 0.7 # Drop columns with more than 70% missing values


df = data.copy()