# --- Initial Data Cleaning (before splitting X and y) ---
print("\n--- Initial Data Cleaning ---")

# Replace empty strings/whitespace with NaN across the entire DataFrame
df = df.replace(r'^\s*$', np.nan, regex=True)

# Explicitly convert potentially mixed-type numeric columns to numeric, coercing errors
numeric_cols_to_coerce = [
    'RID', 'AGE', 'PTEDUCAT', 'APOE4', 'FDG', 'PIB', 'AV45', 'ABETA', 'TAU', 'PTAU',
    'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning',
    'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR',
    'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan',
    'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan',
    'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal', 'Ventricles', 'Hippocampus', 'WholeBrain',
    'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'mPACCdigit', 'mPACCtrailsB',
    'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 'MMSE_bl', 'RAVLT_immediate_bl',
    'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'LDELTOTAL_BL',
    'DIGITSCOR_bl', 'TRABSCOR_bl', 'FAQ_bl', 'mPACCdigit_bl', 'mPACCtrailsB_bl',
    'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl',
    'MidTemp_bl', 'ICV_bl', 'MOCA_bl', 'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl',
    'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl', 'EcogSPMem_bl',
    'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl', 'EcogSPOrgan_bl', 'EcogSPDivatt_bl',
    'EcogSPTotal_bl', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'FDG_bl', 'PIB_bl', 'AV45_bl',
    'Years_bl', 'Month_bl', 'Month', 'M'
]

for col in numeric_cols_to_coerce:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop columns with a high percentage of missing values
cols_to_drop_high_missing = df.columns[df.isnull().sum() / len(df) > MISSING_VALUE_THRESHOLD]
df.drop(columns=cols_to_drop_high_missing, inplace=True)
print(f"Dropped {len(cols_to_drop_high_missing)} columns with > {MISSING_VALUE_THRESHOLD*100}% missing values.")
print(f"Columns remaining: {df.shape[1]}")

# Drop identifier/irrelevant columns before splitting features and target
cols_to_exclude_from_features = [
    'RID', 'PTID', 'VISCODE', 'SITE', 'COLPROT', 'ORIGPROT', 'EXAMDATE',
    'EXAMDATE_bl', 'FLDSTRENG', 'FSVERSION', 'IMAGEUID', 'update_stamp'
]
df = df.drop(columns=[col for col in cols_to_exclude_from_features if col in df.columns], errors='ignore')

# Separate features (X) and target (y)
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# --- Encode Target Variable ---
print("\n--- Encoding Target Variable ---")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))
print(f"Target variable '{TARGET_COLUMN}' encoded mapping: {list(label_encoder.classes_)} to {list(range(len(label_encoder.classes_)))}")
