# --- Function to Run a Model Pipeline ---
def run_model_pipeline(classifier_model, model_name, X_data, y_data_encoded, label_encoder_obj, use_scaler=False):
    """
    Runs the full ML pipeline for a given classifier.

    Args:
        classifier_model: The scikit-learn classifier instance (e.g., RandomForestClassifier()).
        model_name (str): Name of the model for logging and saving.
        X_data (pd.DataFrame): The features DataFrame.
        y_data_encoded (np.array): The encoded target array.
        label_encoder_obj (LabelEncoder): The fitted LabelEncoder object.
        use_scaler (bool): Whether to include StandardScaler in the numerical preprocessing pipeline.
    """
    print(f"\n{'='*50}\n--- Running {model_name} Model ---")

    numerical_features = X_data.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_data.select_dtypes(exclude=np.number).columns.tolist()

    # Define preprocessing for numerical features (median imputation, with optional scaling)
    num_pipeline_steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_scaler:
        num_pipeline_steps.append(('scaler', StandardScaler()))
    numerical_transformer = Pipeline(steps=num_pipeline_steps)

    # Define preprocessing for categorical features (mode imputation + one-hot encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any) as they are
    )

    # Define the final pipeline: preprocessor -> feature selection -> classifier
    # Using SelectFromModel for automated feature selection based on Random Forest importance
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), # Estimator for feature importance
            threshold='median', # Select features with importance > median importance
            prefit=False
        )),
        ('classifier', classifier_model)
    ])

    print("Feature selection method: SelectFromModel using RandomForestClassifier with threshold='median'")

    # Train-Test Split
    print("\n--- Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Model Training
    print(f"\n--- Training {model_name} ---")
    final_pipeline.fit(X_train, y_train)
    print(f"{model_name} training complete.")

    # --- Feature Selection Output ---
    print("\n--- Feature Selection Output ---")
    # Get feature names after preprocessing by the ColumnTransformer
    # We need to transform a small part of X_train to get the correct shape for feature names
    # For compatibility, we'll manually combine numerical and one-hot encoded names.

    # Get names of numerical features that went into the preprocessor
    num_features_in = numerical_features

    # Get names of one-hot encoded categorical features that went into the preprocessor
    cat_ohe_step = final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features_out = list(cat_ohe_step.get_feature_names_out(categorical_features))

    # Combine all feature names that come out of the preprocessor and go into SelectFromModel
    all_features_after_preprocessing = num_features_in + cat_features_out

    # Get the support mask from SelectFromModel
    selected_features_mask = final_pipeline.named_steps['feature_selection'].get_support()

    # Get the names of the selected features
    selected_feature_names = [all_features_after_preprocessing[i] for i, selected in enumerate(selected_features_mask) if selected]

    print(f"Number of features selected by SelectFromModel: {len(selected_feature_names)}")
    print("Selected Features:")
    for feature in selected_feature_names:
        print(f"- {feature}")


    # Model Evaluation
    print(f"\n--- {model_name} Evaluation ---")
    y_pred = final_pipeline.predict(X_test)

    # Confusion Matrix (textual output)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder_obj.classes_,
                yticklabels=label_encoder_obj.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    # Classification Report (Accuracy, Precision, Recall, F1-score)
    print("\nClassification Report (Accuracy, Precision, Recall, F1-score):")
    print(classification_report(y_test, y_pred, target_names=label_encoder_obj.classes_))

    # ROC AUC Score
    if len(label_encoder_obj.classes_) > 2:
        try:
            y_proba = final_pipeline.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            print(f"ROC AUC Score (One-vs-Rest, Weighted): {roc_auc:.4f}")
        except ValueError as e:
            print(f"Could not calculate ROC AUC score: {e}")
            print("This might happen if there's only one class present in y_true or y_score for a binary case, or if classes are missing in one of the sets.")
    else:
        # Binary classification case
        roc_auc = roc_auc_score(y_test, final_pipeline.predict_proba(X_test)[:, 1])
        print(f"ROC AUC Score: {roc_auc:.4f}")

    # Save the Model
    model_save_path = f"alzheimer_prediction_model_{model_name.lower().replace(' ', '_')}.pkl"
    print(f"\n--- Saving {model_name} Model ---")
    try:
        joblib.dump(final_pipeline, model_save_path)
        print(f"Trained {model_name} model (full pipeline) saved to '{model_save_path}'")
    except Exception as e:
        print(f"Error saving the {model_name} model: {e}")

    print(f"{'='*50}\n")