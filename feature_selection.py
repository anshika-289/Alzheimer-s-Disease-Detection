def get_preprocessor_and_selector(X_data, use_scaler):
    numerical_features = X_data.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_data.select_dtypes(exclude=np.number).columns.tolist()

    num_pipeline_steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_scaler:
        num_pipeline_steps.append(('scaler', StandardScaler()))
    numerical_transformer = Pipeline(steps=num_pipeline_steps)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        threshold='median',
        prefit=False
    )

    return preprocessor, selector, numerical_features, categorical_features
