import json
import datetime
import pytz
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, from_unixtime, count, avg

def init_spark(memory="4g"):
    """
    Initialize Spark session with optimized configuration for large data processing
    """
    spark = SparkSession.builder \
        .appName("TwitterAnalysis") \
        .master("local[*]") \
        .config("spark.driver.memory", memory) \
        .config("spark.executor.memory", "4g") \
        .config("spark.memory.fraction", 0.8) \
        .config("spark.memory.storageFraction", 0.3) \
        .config("spark.sql.shuffle.partitions", 8) \
        .config("spark.default.parallelism", 8) \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.python.worker.memory", "2g") \
        .config("spark.python.worker.timeout", 1800) \
        .config("spark.local.dir", "./spark-temp") \
        .getOrCreate()
    
    # 设置日志级别以减少输出
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_tweet_data(spark, hashtag, sample_size=None):
    """
    Load tweet data for a specific hashtag using Spark
    
    Parameters:
    -----------
    spark : SparkSession
        Initialized Spark session
    hashtag : str
        Hashtag to load data for (without the # symbol)
    sample_size : int, optional
        Number of tweets to sample (None means load all)
        
    Returns:
    --------
    df : DataFrame
        Spark DataFrame containing the loaded tweets
    """
    file_path = f"ECE219_tweet_data/tweets_#{hashtag.lower()}.txt"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # 直接使用read.json读取，这样更高效
        df = spark.read.json(file_path)
        
        if sample_size:
            # 使用limit而不是sample，这样更轻量级
            df = df.limit(sample_size)
        
        # 添加datetime列以便后续处理
        df = df.withColumn("datetime", from_unixtime("citation_date"))
        df = df.withColumn("hour", hour("datetime"))
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def convert_unix_to_datetime(unix_time, timezone=None):
    """
    Convert UNIX timestamp to datetime object
    
    Parameters:
    -----------
    unix_time : float
        UNIX timestamp
    timezone : str, optional
        Timezone to convert to (e.g., 'America/Los_Angeles')
        
    Returns:
    --------
    datetime_obj : datetime
        Datetime object corresponding to the UNIX timestamp
    """
    if timezone:
        tz = pytz.timezone(timezone)
        datetime_obj = datetime.datetime.fromtimestamp(unix_time, tz)
    else:
        datetime_obj = datetime.datetime.fromtimestamp(unix_time)
    
    return datetime_obj

def get_retweet_count(tweet_json):
    """
    Extract retweet count from tweet JSON
    
    Parameters:
    -----------
    tweet_json : dict
        Tweet JSON object
        
    Returns:
    --------
    retweet_count : int
        Number of retweets
    """
    try:
        return tweet_json['metrics']['citations']['total']
    except (KeyError, TypeError):
        return 0

def get_followers_count(tweet_json):
    """
    Extract followers count from tweet JSON
    
    Parameters:
    -----------
    tweet_json : dict
        Tweet JSON object
        
    Returns:
    --------
    followers_count : int
        Number of followers
    """
    try:
        return tweet_json['author']['followers']
    except (KeyError, TypeError):
        return 0 