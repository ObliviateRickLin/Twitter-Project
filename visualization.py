import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

def plot_tweet_frequency(df, time_col="hour", save_path=None):
    """
    Plot tweet frequency over time
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame containing tweet data
    time_col : str
        Name of the time column to plot
    save_path : str, optional
        Path to save the plot to
    """
    # Group by time column
    freq_df = df.groupby(time_col).size().reset_index(name="count")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(freq_df[time_col], freq_df["count"], alpha=0.7)
    plt.title("Tweet Frequency")
    plt.xlabel(time_col.capitalize())
    plt.ylabel("Number of Tweets")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Tweet frequency plot saved to {save_path}")
    
    plt.close()
    
    return freq_df

def plot_user_distribution(df, column="follower_count", bins=50, log_scale=True, save_path=None):
    """
    Plot distribution of user-related metrics
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame containing user data
    column : str
        Column to plot
    bins : int
        Number of bins for histogram
    log_scale : bool
        Whether to use log scale for y-axis
    save_path : str, optional
        Path to save the plot to
    """
    # Plot
    plt.figure(figsize=(12, 6))
    plt.hist(df[column], bins=bins, alpha=0.7)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column.capitalize().replace("_", " "))
    plt.ylabel("Number of Users")
    
    if log_scale:
        plt.yscale("log")
    
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"User distribution plot saved to {save_path}")
    
    plt.close()

def plot_engagement_metrics(df, metrics=None, save_path=None):
    """
    Plot engagement metrics
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame containing engagement metrics
    metrics : list, optional
        List of metrics to plot
    save_path : str, optional
        Path to save the plot to
    """
    if metrics is None:
        metrics = ["retweet_count", "reply_count", "like_count"]
    
    # Filter out metrics not in DataFrame
    metrics = [m for m in metrics if m in df.columns]
    
    if not metrics:
        print("No valid metrics to plot")
        return
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        plt.hist(df[metric], bins=50, alpha=0.7)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric.capitalize().replace("_", " "))
        plt.ylabel("Count")
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Engagement metrics plot saved to {save_path}")
    
    plt.close()

def plot_correlation_matrix(df, columns=None, save_path=None):
    """
    Plot correlation matrix
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame containing features
    columns : list, optional
        List of columns to include in correlation matrix
    save_path : str, optional
        Path to save the plot to
    """
    # Select columns
    if columns is not None:
        df = df[columns]
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#4169E1", "white", "#FF6347"])
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap=cmap, center=0, fmt=".2f")
    plt.title("Correlation Matrix")
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Correlation matrix plot saved to {save_path}")
    
    plt.close()
    
    return corr

def plot_wordcloud(text_series, stopwords=None, save_path=None):
    """
    Plot word cloud
    
    Parameters:
    -----------
    text_series : Series
        Pandas Series containing text
    stopwords : set, optional
        Set of stopwords to exclude
    save_path : str, optional
        Path to save the plot to
    """
    # Combine all text
    text = " ".join(text_series.astype(str))
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white", 
        stopwords=stopwords,
        max_words=100
    ).generate(text)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Word cloud plot saved to {save_path}")
    
    plt.close()

def plot_scatter(df, x_col, y_col, hue_col=None, size_col=None, alpha=0.7, save_path=None):
    """
    Plot scatter plot
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame containing data
    x_col : str
        Name of x-axis column
    y_col : str
        Name of y-axis column
    hue_col : str, optional
        Name of column to use for point color
    size_col : str, optional
        Name of column to use for point size
    alpha : float
        Transparency of points
    save_path : str, optional
        Path to save the plot to
    """
    # Plot
    plt.figure(figsize=(12, 8))
    
    if hue_col and size_col:
        scatter = plt.scatter(
            df[x_col], 
            df[y_col], 
            c=df[hue_col],
            s=df[size_col],
            alpha=alpha, 
            cmap="viridis"
        )
        plt.colorbar(scatter, label=hue_col)
    elif hue_col:
        scatter = plt.scatter(
            df[x_col], 
            df[y_col], 
            c=df[hue_col],
            alpha=alpha, 
            cmap="viridis"
        )
        plt.colorbar(scatter, label=hue_col)
    elif size_col:
        plt.scatter(
            df[x_col], 
            df[y_col], 
            s=df[size_col],
            alpha=alpha
        )
    else:
        plt.scatter(
            df[x_col], 
            df[y_col], 
            alpha=alpha
        )
    
    plt.title(f"{y_col} vs {x_col}")
    plt.xlabel(x_col.capitalize().replace("_", " "))
    plt.ylabel(y_col.capitalize().replace("_", " "))
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Scatter plot saved to {save_path}")
    
    plt.close()

def plot_confusion_matrix(cm, classes=None, normalize=False, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : array
        Confusion matrix
    classes : list, optional
        List of class labels
    normalize : bool
        Whether to normalize the confusion matrix
    save_path : str, optional
        Path to save the plot to
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm, 
        annot=True, 
        cmap="Blues", 
        fmt=".2f" if normalize else "d",
        xticklabels=classes,
        yticklabels=classes
    )
    
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.close() 