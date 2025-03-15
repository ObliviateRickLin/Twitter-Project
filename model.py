import os
import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def prepare_features(df, feature_cols, target_col, vector_col="features"):
    """
    Prepare features for modeling
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame containing features
    feature_cols : list
        List of feature column names
    target_col : str
        Name of the target column
    vector_col : str
        Name of the output vector column
        
    Returns:
    --------
    df : DataFrame
        DataFrame with assembled features
    """
    # Assemble features into a single vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=vector_col)
    df = assembler.transform(df)
    
    return df

def split_data(df, train_ratio=0.8, seed=42):
    """
    Split data into training and test sets
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame containing features and target
    train_ratio : float
        Ratio of training data
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_df : DataFrame
        Training DataFrame
    test_df : DataFrame
        Test DataFrame
    """
    # Split the data
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    
    return train_df, test_df

def train_baseline_classifier(train_df, feature_col="features", target_col="label"):
    """
    Train a baseline logistic regression classifier
    
    Parameters:
    -----------
    train_df : DataFrame
        Training DataFrame
    feature_col : str
        Name of the feature column
    target_col : str
        Name of the target column
        
    Returns:
    --------
    model : LogisticRegressionModel
        Trained logistic regression model
    """
    # Create and train a logistic regression model
    lr = LogisticRegression(featuresCol=feature_col, labelCol=target_col, maxIter=10)
    model = lr.fit(train_df)
    
    return model

def train_baseline_regressor(train_df, feature_col="features", target_col="label"):
    """
    Train a baseline linear regression model
    
    Parameters:
    -----------
    train_df : DataFrame
        Training DataFrame
    feature_col : str
        Name of the feature column
    target_col : str
        Name of the target column
        
    Returns:
    --------
    model : LinearRegressionModel
        Trained linear regression model
    """
    # Create and train a linear regression model
    lr = LinearRegression(featuresCol=feature_col, labelCol=target_col, maxIter=10)
    model = lr.fit(train_df)
    
    return model

def train_advanced_classifier(train_df, feature_col="features", target_col="label", cv_folds=3):
    """
    Train an advanced random forest classifier with cross-validation
    
    Parameters:
    -----------
    train_df : DataFrame
        Training DataFrame
    feature_col : str
        Name of the feature column
    target_col : str
        Name of the target column
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    best_model : RandomForestClassificationModel
        Best trained random forest model
    """
    # Create a random forest classifier
    rf = RandomForestClassifier(featuresCol=feature_col, labelCol=target_col, seed=42)
    
    # Create a parameter grid for tuning
    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 30]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()
    
    # Create a multi-class evaluator
    evaluator = MulticlassClassificationEvaluator(
        labelCol=target_col, 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    
    # Create a cross-validator
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=cv_folds
    )
    
    # Train the model using cross-validation
    cv_model = cv.fit(train_df)
    
    # Get the best model
    best_model = cv_model.bestModel
    
    return best_model, cv_model

def train_advanced_regressor(train_df, feature_col="features", target_col="label", cv_folds=3):
    """
    Train an advanced random forest regressor with cross-validation
    
    Parameters:
    -----------
    train_df : DataFrame
        Training DataFrame
    feature_col : str
        Name of the feature column
    target_col : str
        Name of the target column
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    best_model : RandomForestRegressionModel
        Best trained random forest model
    """
    # Create a random forest regressor
    rf = RandomForestRegressor(featuresCol=feature_col, labelCol=target_col, seed=42)
    
    # Create a parameter grid for tuning
    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 30]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()
    
    # Create a regression evaluator
    evaluator = RegressionEvaluator(
        labelCol=target_col, 
        predictionCol="prediction", 
        metricName="rmse"
    )
    
    # Create a cross-validator
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=cv_folds
    )
    
    # Train the model using cross-validation
    cv_model = cv.fit(train_df)
    
    # Get the best model
    best_model = cv_model.bestModel
    
    return best_model, cv_model

def evaluate_classifier(model, test_df, target_col="label", prediction_col="prediction"):
    """
    Evaluate a classification model
    
    Parameters:
    -----------
    model : Model
        Trained classification model
    test_df : DataFrame
        Test DataFrame
    target_col : str
        Name of the target column
    prediction_col : str
        Name of the prediction column
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    predictions = model.transform(test_df)
    
    # Create evaluators for different metrics
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol=target_col, 
        predictionCol=prediction_col, 
        metricName="accuracy"
    )
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=target_col, 
        predictionCol=prediction_col, 
        metricName="f1"
    )
    
    # Calculate metrics
    accuracy = evaluator_accuracy.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    
    # Convert to Pandas for detailed metrics
    predictions_pd = predictions.select(target_col, prediction_col).toPandas()
    
    # Get confusion matrix and classification report
    cm = confusion_matrix(predictions_pd[target_col], predictions_pd[prediction_col])
    report = classification_report(predictions_pd[target_col], predictions_pd[prediction_col], output_dict=True)
    
    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    return metrics, predictions

def evaluate_regressor(model, test_df, target_col="label", prediction_col="prediction"):
    """
    Evaluate a regression model
    
    Parameters:
    -----------
    model : Model
        Trained regression model
    test_df : DataFrame
        Test DataFrame
    target_col : str
        Name of the target column
    prediction_col : str
        Name of the prediction column
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    predictions = model.transform(test_df)
    
    # Create evaluators for different metrics
    evaluator_rmse = RegressionEvaluator(
        labelCol=target_col, 
        predictionCol=prediction_col, 
        metricName="rmse"
    )
    
    evaluator_r2 = RegressionEvaluator(
        labelCol=target_col, 
        predictionCol=prediction_col, 
        metricName="r2"
    )
    
    evaluator_mae = RegressionEvaluator(
        labelCol=target_col, 
        predictionCol=prediction_col, 
        metricName="mae"
    )
    
    # Calculate metrics
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    
    # Create metrics dictionary
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    
    return metrics, predictions

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : Model
        Trained tree-based model
    feature_names : list
        List of feature names
    save_path : str, optional
        Path to save the plot to
    """
    # Get feature importance
    if hasattr(model, "featureImportances"):
        importances = model.featureImportances.toArray()
        
        # Create a DataFrame for plotting
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values("Importance", ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df["Feature"][:20], importance_df["Importance"][:20])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Save the plot if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        
        plt.close()
        
        return importance_df
    else:
        print("Model does not have feature importances attribute.")

def create_multitask_features(df):
    """
    Create features for multitask learning
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame with all features
        
    Returns:
    --------
    df : DataFrame
        DataFrame with features prepared for different tasks
    """
    # 1. 球队归属预测特征
    team_features = [
        "text_features",
        "patriots_relevance",
        "seahawks_relevance",
        "game_period",
        "game_quarter",
        "hours_from_game_start"
    ]
    
    # 2. 影响力预测特征
    influence_features = [
        "text_features",
        "follower_count",
        "following_count",
        "is_verified",
        "game_period",
        "part_of_day",
        "is_weekend"
    ]
    
    # 3. 时间段预测特征
    time_features = [
        "text_features",
        "hour_of_day",
        "day_of_week",
        "patriots_relevance",
        "seahawks_relevance"
    ]
    
    # 为每个任务创建特征向量
    for feature_set, name in [
        (team_features, "team_features"),
        (influence_features, "influence_features"),
        (time_features, "time_features")
    ]:
        assembler = VectorAssembler(
            inputCols=[f for f in feature_set if f in df.columns],
            outputCol=f"{name}_vector"
        )
        df = assembler.transform(df)
    
    return df

def train_multitask_model(train_df, cv_folds=3):
    """
    Train models for multiple prediction tasks
    
    Parameters:
    -----------
    train_df : DataFrame
        Training DataFrame with prepared features
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    models : dict
        Dictionary containing trained models for each task
    """
    # 1. 球队归属预测（分类）
    team_classifier = RandomForestClassifier(
        featuresCol="team_features_vector",
        labelCol="team_label",
        numTrees=100,
        maxDepth=10
    )
    
    # 2. 影响力预测（回归）
    influence_regressor = RandomForestRegressor(
        featuresCol="influence_features_vector",
        labelCol="influence_score",
        numTrees=100,
        maxDepth=10
    )
    
    # 3. 时间段预测（分类）
    time_classifier = RandomForestClassifier(
        featuresCol="time_features_vector",
        labelCol="time_period",
        numTrees=100,
        maxDepth=10
    )
    
    # 创建评估器
    classif_evaluator = MulticlassClassificationEvaluator(metricName="f1")
    regr_evaluator = RegressionEvaluator(metricName="rmse")
    
    # 训练模型
    models = {}
    for model, name, evaluator in [
        (team_classifier, "team", classif_evaluator),
        (influence_regressor, "influence", regr_evaluator),
        (time_classifier, "time", classif_evaluator)
    ]:
        # 创建参数网格
        param_grid = ParamGridBuilder() \
            .addGrid(model.numTrees, [50, 100]) \
            .addGrid(model.maxDepth, [5, 10]) \
            .build()
        
        # 创建交叉验证
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=cv_folds
        )
        
        # 训练并保存最佳模型
        cv_model = cv.fit(train_df)
        models[name] = cv_model.bestModel
    
    return models

def evaluate_multitask_model(models, test_df):
    """
    Evaluate models for all tasks
    
    Parameters:
    -----------
    models : dict
        Dictionary containing trained models
    test_df : DataFrame
        Test DataFrame
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics for each task
    """
    metrics = {}
    
    # 1. 评估球队归属预测
    team_preds = models["team"].transform(test_df)
    metrics["team"] = {
        "accuracy": MulticlassClassificationEvaluator(
            labelCol="team_label",
            predictionCol="prediction",
            metricName="accuracy"
        ).evaluate(team_preds),
        "f1": MulticlassClassificationEvaluator(
            labelCol="team_label",
            predictionCol="prediction",
            metricName="f1"
        ).evaluate(team_preds)
    }
    
    # 2. 评估影响力预测
    influence_preds = models["influence"].transform(test_df)
    metrics["influence"] = {
        "rmse": RegressionEvaluator(
            labelCol="influence_score",
            predictionCol="prediction",
            metricName="rmse"
        ).evaluate(influence_preds),
        "r2": RegressionEvaluator(
            labelCol="influence_score",
            predictionCol="prediction",
            metricName="r2"
        ).evaluate(influence_preds)
    }
    
    # 3. 评估时间段预测
    time_preds = models["time"].transform(test_df)
    metrics["time"] = {
        "accuracy": MulticlassClassificationEvaluator(
            labelCol="time_period",
            predictionCol="prediction",
            metricName="accuracy"
        ).evaluate(time_preds),
        "f1": MulticlassClassificationEvaluator(
            labelCol="time_period",
            predictionCol="prediction",
            metricName="f1"
        ).evaluate(time_preds)
    }
    
    return metrics 