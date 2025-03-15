import os
import pandas as pd
import numpy as np
from pyspark.sql.functions import hour, from_unixtime, count, avg, col
import matplotlib.pyplot as plt
from data_utils import init_spark, load_tweet_data

def calculate_basic_stats(spark, hashtag, sample_size=None):
    """
    Calculate basic statistics for a given hashtag
    
    Parameters:
    -----------
    spark : SparkSession
        Initialized Spark session
    hashtag : str
        Hashtag to analyze (without the # symbol)
    sample_size : int, optional
        Number of tweets to sample
        
    Returns:
    --------
    stats : dict
        Dictionary containing the calculated statistics
    """
    print(f"\nAnalyzing #{hashtag}...")
    
    try:
        # Load the data
        df = load_tweet_data(spark, hashtag, sample_size)
        
        # 1. Calculate hourly tweet counts
        hourly_counts = df.groupBy("hour").count()
        
        # Get total hours and tweets for average calculation
        total_hours = hourly_counts.count()
        total_tweets = df.count()
        avg_tweets_per_hour = total_tweets / total_hours if total_hours > 0 else 0
        
        # 2. Calculate average followers per tweet
        avg_followers = df.select(avg(col("author.followers"))).collect()[0][0]
        
        # 3. Calculate average retweets per tweet
        avg_retweets = df.select(avg(col("metrics.citations.total"))).collect()[0][0]
        
        # Print statistics
        print(f"\nStatistics for #{hashtag}:")
        print(f"Total tweets analyzed: {total_tweets}")
        print(f"Average tweets per hour: {avg_tweets_per_hour:.2f}")
        print(f"Average followers per tweet: {avg_followers:.2f}")
        print(f"Average retweets per tweet: {avg_retweets:.2f}")
        print("=" * 50)
        
        # Save hourly counts to CSV for plotting
        os.makedirs("analysis/stats", exist_ok=True)
        hourly_counts_pd = hourly_counts.toPandas()
        hourly_counts_pd.to_csv(f"analysis/stats/{hashtag}_hourly_counts.csv", index=False)
        
        # Create stats dictionary
        stats = {
            "hashtag": hashtag,
            "total_tweets": total_tweets,
            "avg_tweets_per_hour": avg_tweets_per_hour,
            "avg_followers_per_tweet": avg_followers,
            "avg_retweets_per_tweet": avg_retweets,
            "hourly_counts": hourly_counts_pd
        }
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing #{hashtag}: {str(e)}")
        return None

def plot_hourly_tweets(hashtag_stats, save_path=None):
    """
    Plot the number of tweets per hour for a given hashtag
    
    Parameters:
    -----------
    hashtag_stats : dict
        Dictionary containing the hashtag statistics
    save_path : str, optional
        Path to save the plot to
    """
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Sort by hour
    df = hashtag_stats["hourly_counts"].sort_values("hour")
    
    # Plot
    plt.bar(df["hour"], df["count"], alpha=0.7)
    plt.title(f"Number of Tweets per Hour for #{hashtag_stats['hashtag']}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Tweets")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(range(0, 24))
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def compare_hourly_tweets(hashtag_stats_list, save_path=None):
    """
    Compare the number of tweets per hour for multiple hashtags
    
    Parameters:
    -----------
    hashtag_stats_list : list
        List of dictionaries containing hashtag statistics
    save_path : str, optional
        Path to save the plot to
    """
    # Create the plot
    plt.figure(figsize=(14, 7))
    
    # Plot each hashtag
    for stats in hashtag_stats_list:
        df = stats["hourly_counts"].sort_values("hour")
        plt.plot(df["hour"], df["count"], marker="o", label=f"#{stats['hashtag']}")
    
    plt.title("Number of Tweets per Hour - Comparison")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Tweets")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(range(0, 24))
    plt.legend()
    
    # Save the plot if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()

def run_basic_stats_analysis(sample_size=None):
    """
    Run basic statistics analysis for all hashtags
    
    Parameters:
    -----------
    sample_size : int, optional
        Number of tweets to sample for each hashtag
    
    Returns:
    --------
    all_stats : dict
        Dictionary containing statistics for all hashtags
    """
    # Initialize Spark
    spark = init_spark()
    
    # List of hashtags to analyze
    hashtags = ["SuperBowl", "NFL", "patriots", "gopatriots", "gohawks", "sb49"]
    
    # Calculate statistics for each hashtag
    all_stats = {}
    for hashtag in hashtags:
        try:
            stats = calculate_basic_stats(spark, hashtag, sample_size)
            all_stats[hashtag] = stats
            
            # Generate individual plots
            plot_hourly_tweets(
                stats, 
                save_path=f"analysis/figures/{hashtag}_hourly_tweets.png"
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    # Compare SuperBowl and NFL as required in the assignment
    if "SuperBowl" in all_stats and "NFL" in all_stats:
        compare_hourly_tweets(
            [all_stats["SuperBowl"], all_stats["NFL"]],
            save_path="analysis/figures/SuperBowl_NFL_comparison.png"
        )
    
    # Stop the Spark session
    spark.stop()
    
    return all_stats

if __name__ == "__main__":
    # Run with a sample size to test (set to None to use all data)
    run_basic_stats_analysis(sample_size=10000) 