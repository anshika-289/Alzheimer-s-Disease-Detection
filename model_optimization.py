
# --- Define Parameter Grids for Hyperparameter Tuning ---
logistic_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs'] # Add 'lbfgs' for multinomial logistic regression
}

rf_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10]
}


# --- Function to Run a Model Pipeline ---
def run_model_pipeline(classifier_model_instance, model_name, X_data, y_data_encoded, label_encoder_obj, use_scaler=False, param_grid=None):
    """
    Runs the full ML pipeline for a given classifier, with optional hyperparameter tuning.

    Args:
        classifier_model_instance: The scikit-learn classifier instance (e.g., RandomForestClassifier()).
        model_name (str): Name of the model for logging and saving.
        X_data (pd.DataFrame): The features DataFrame.
        y_data_encoded (np.array): The encoded target array.
        label_encoder_obj (LabelEncoder): The fitted LabelEncoder object.
        use_scaler (bool): Whether to include StandardScaler in the numerical preprocessing pipeline.
        param_grid (dict or list of dict, optional): Hyperparameter grid for GridSearchCV.
                                                    If provided, tuning will be performed.
                                                    Parameters should be prefixed with 'classifier__'.
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
        remainder='passthrough'
    )

    # Create the full pipeline including preprocessing and the classifier
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifier_model_instance)])

    # Train-Test Split
    print("\n--- Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_data_encoded)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Run hyperparameter tuning if param_grid is provided
    if param_grid:
        print("\n--- Running GridSearchCV for Hyperparameter Tuning ---")
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print("\nBest parameters found:")
        print(grid_search.best_params_)
        print("\nBest cross-validation accuracy:")
        print(grid_search.best_score_)

        # Use the best model found by GridSearchCV
        best_model = grid_search.best_estimator_
        print(f"\n--- Evaluating Best {model_name} Model on Test Set ---")
        y_pred = best_model.predict(X_test)

        # Decode predictions for reporting
        y_pred_decoded = label_encoder_obj.inverse_transform(y_pred)
        y_test_decoded = label_encoder_obj.inverse_transform(y_test)

        # Evaluation Metrics
        print("\nClassification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded))

        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)
        print(cm)

        # You can also plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder_obj.classes_, yticklabels=label_encoder_obj.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

        # AUC-ROC Score (for multi-class, needs one-vs-rest or one-vs-one)
        # Simplest is average of one-vs-rest
        try:
            y_prob = best_model.predict_proba(X_test)
            auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')
            print(f"\nAUC-ROC Score (One-vs-Rest): {auc_score:.4f}")
        except ValueError as e:
             print(f"\nCould not compute AUC-ROC score: {e}")
             print("AUC-ROC is typically for binary classification or needs specific handling for multi-class.")


    else:
        # Train the pipeline with the provided classifier if no param_grid
        print(f"\n--- Training {model_name} Model ---")
        pipeline.fit(X_train, y_train)

        print(f"\n--- Evaluating {model_name} Model on Test Set ---")
        y_pred = pipeline.predict(X_test)

        # Decode predictions for reporting
        y_pred_decoded = label_encoder_obj.inverse_transform(y_pred)
        y_test_decoded = label_encoder_obj.inverse_transform(y_test)

        # Evaluation Metrics
        print("\nClassification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded))

        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)
        print(cm)

        # You can also plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder_obj.classes_, yticklabels=label_encoder_obj.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()
