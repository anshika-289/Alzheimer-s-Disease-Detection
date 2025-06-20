# --- Run Logistic Regression ---
run_model_pipeline(
    classifier_model=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial', solver='lbfgs', class_weight='balanced'),
    model_name="Logistic Regression",
    X_data=X,
    y_data_encoded=y_encoded,
    label_encoder_obj=label_encoder,
    use_scaler=True # Apply StandardScaler for Logistic Regression
)

# --- Run Random Forest ---
run_model_pipeline(
    classifier_model=RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced'),
    model_name="Random Forest",
    X_data=X,
    y_data_encoded=y_encoded,
    label_encoder_obj=label_encoder,
    use_scaler=False # No StandardScaler needed for Random Forest
)

# --- Run Support Vector Machine (SVM) ---
run_model_pipeline(
    classifier_model=SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, class_weight='balanced'), # RBF kernel, enable probability for AUC-ROC
    model_name="SVM",
    X_data=X,
    y_data_encoded=y_encoded,
    label_encoder_obj=label_encoder,
    use_scaler=True # SVMs are sensitive to feature scaling
)