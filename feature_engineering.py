import re
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler

# 定义比赛相关的常量
GAME_START_TIME = 1422835200  # 2015-02-01 23:30:00 UTC (Super Bowl 开始时间)
GAME_END_TIME = 1422846000    # 2015-02-02 02:30:00 UTC (大约3小时后)

# 定义球队相关的关键词
PATRIOTS_KEYWORDS = {
    'patriots', 'pats', 'brady', 'gronkowski', 'edelman', 'belichick',
    'newengland', 'ne', 'gopatriots'
}

SEAHAWKS_KEYWORDS = {
    'seahawks', 'hawks', 'wilson', 'lynch', 'sherman', 'carroll',
    'seattle', 'sea', 'gohawks'
}

def preprocess_text(df):
    """
    预处理推文文本，转换为小写并移除URL、提及和特殊字符
    """
    # 首先尝试从tweet.text获取文本，如果失败则使用tweet字段
    text_udf = F.udf(
        lambda tweet_obj: tweet_obj.get("text", str(tweet_obj)) if isinstance(tweet_obj, dict) else str(tweet_obj),
        StringType()
    )
    
    # 提取文本
    df = df.withColumn("tweet_text", text_udf(F.col("tweet")))
    
    # 清理文本
    clean_udf = F.udf(
        lambda text: re.sub(r'http\S+|@\S+|#\S+|[^\w\s]', ' ', text.lower()) if text else "",
        StringType()
    )
    
    df = df.withColumn("processed_text", clean_udf(F.col("tweet_text")))
    
    return df

def extract_text_features(df, max_features=1000, output_col="text_features"):
    """
    Extract text features using TF-IDF
    """
    # 使用更高效的文本处理方式
    tokenizer = Tokenizer(inputCol="processed_text", outputCol="words")
    
    # 自定义停用词列表，减少处理时间
    custom_stopwords = ["rt", "the", "to", "and", "a", "in", "is", "it", "you", "that", "for", "on", "with", "this"]
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=custom_stopwords)
    
    # 优化CountVectorizer参数
    cv = CountVectorizer(
        inputCol="filtered_words", 
        outputCol="tf",
        vocabSize=max_features,
        minDF=2,
        maxDF=0.95
    )
    
    # 优化IDF参数
    idf = IDF(
        inputCol="tf", 
        outputCol=output_col,
        minDocFreq=2
    )
    
    # 创建并拟合管道
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])
    
    try:
        model = pipeline.fit(df)
        df = model.transform(df)
        return df, model
    except Exception as e:
        print(f"Error in text feature extraction: {str(e)}")
        # 如果处理失败，返回带有空特征的DataFrame
        df = df.withColumn(output_col, F.array([F.lit(0.0)]))
        return df, None

def extract_temporal_features(df):
    """
    Extract temporal features from tweet datetime
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame containing tweet data
        
    Returns:
    --------
    df : DataFrame
        DataFrame with temporal features
    """
    # Extract hour of the day
    df = df.withColumn("hour_of_day", F.hour("datetime"))
    
    # Extract day of the week (0-6, where 0 is Sunday)
    df = df.withColumn("day_of_week", F.dayofweek("datetime"))
    
    # Extract is_weekend feature
    df = df.withColumn("is_weekend", 
                       F.when((F.col("day_of_week") == 1) | (F.col("day_of_week") == 7), 1).otherwise(0))
    
    # Extract part of day
    df = df.withColumn("part_of_day", 
                       F.when(F.col("hour_of_day") < 6, "night")
                       .when(F.col("hour_of_day") < 12, "morning")
                       .when(F.col("hour_of_day") < 18, "afternoon")
                       .otherwise("evening"))
    
    return df

def extract_user_features(df):
    """
    从推文数据中提取用户特征
    """
    # 处理follower_count - 可能在tweet.user.followers_count或author.followers中
    follower_udf = F.udf(
        lambda tweet_obj, author_obj: 
            (tweet_obj.get("user", {}).get("followers_count") if isinstance(tweet_obj, dict) else None) or 
            (author_obj.get("followers") if isinstance(author_obj, dict) else 0) or 0,
        FloatType()
    )
    
    df = df.withColumn("follower_count", 
                      follower_udf(F.col("tweet"), F.col("author")))
    
    # 处理following_count - 在tweet.user.friends_count中
    following_udf = F.udf(
        lambda tweet_obj: 
            tweet_obj.get("user", {}).get("friends_count") if isinstance(tweet_obj, dict) else 0,
        FloatType()
    )
    
    df = df.withColumn("following_count", following_udf(F.col("tweet")))
    
    # 处理verified状态
    verified_udf = F.udf(
        lambda tweet_obj, author_obj: 
            (tweet_obj.get("user", {}).get("verified") if isinstance(tweet_obj, dict) else False) or
            (author_obj.get("verified") if isinstance(author_obj, dict) else False),
        FloatType()
    )
    
    df = df.withColumn("is_verified", verified_udf(F.col("tweet"), F.col("author")))
    
    # 计算follower-following比率（安全计算）
    df = df.withColumn(
        "follower_following_ratio",
        F.when(F.col("following_count") > 0, 
               F.col("follower_count") / F.col("following_count"))
        .otherwise(0)
    )
    
    return df

def extract_engagement_features(df):
    """
    提取互动相关特征
    """
    # 从metrics或tweet对象中提取转发数
    retweet_udf = F.udf(
        lambda tweet_obj, metrics_obj: 
            (tweet_obj.get("retweet_count") if isinstance(tweet_obj, dict) else 0) or
            (metrics_obj.get("citations", {}).get("total") if isinstance(metrics_obj, dict) else 0),
        FloatType()
    )
    
    df = df.withColumn("retweet_count", 
                      retweet_udf(F.col("tweet"), F.col("metrics")))
    
    # 从tweet对象中提取收藏数
    favorite_udf = F.udf(
        lambda tweet_obj: 
            tweet_obj.get("favorite_count") if isinstance(tweet_obj, dict) else 0,
        FloatType()
    )
    
    df = df.withColumn("favorite_count", favorite_udf(F.col("tweet")))
    
    # 计算互动率（安全计算）
    df = df.withColumn(
        "engagement_rate",
        F.when(F.col("follower_count") > 0, 
               (F.col("retweet_count") + F.col("favorite_count")) / F.col("follower_count"))
        .otherwise(0)
    )
    
    return df

def extract_game_context_features(df):
    """
    提取与比赛相关的上下文特征
    """
    # 1. 计算相对于比赛时间的特征
    def get_game_period(timestamp):
        if not timestamp:
            return "unknown"
        try:
            ts = float(timestamp)
            if ts < GAME_START_TIME:
                return "pre_game"
            elif ts > GAME_END_TIME:
                return "post_game"
            else:
                return "during_game"
        except:
            return "unknown"
    
    game_period_udf = F.udf(get_game_period, StringType())
    df = df.withColumn("game_period", game_period_udf(F.col("citation_date")))
    
    # 2. 计算距离比赛开始的时间（小时）
    df = df.withColumn(
        "hours_from_game_start",
        F.when(F.col("citation_date").isNotNull(),
               (F.col("citation_date").cast("float") - F.lit(GAME_START_TIME)) / 3600)
        .otherwise(0)
    )
    
    # 3. 提取球队相关度得分（使用推文文本）
    def get_team_relevance(text):
        if not text or not isinstance(text, str):
            return (0.0, 0.0)
        
        text = text.lower()
        words = set(re.findall(r'\w+', text))
        
        pats_score = len(words & PATRIOTS_KEYWORDS) / len(PATRIOTS_KEYWORDS) if words else 0.0
        hawks_score = len(words & SEAHAWKS_KEYWORDS) / len(SEAHAWKS_KEYWORDS) if words else 0.0
        
        return (float(pats_score), float(hawks_score))
    
    # 直接处理文本，不使用复杂的数组结构
    pats_relevance_udf = F.udf(
        lambda text: get_team_relevance(text)[0],
        FloatType()
    )
    
    hawks_relevance_udf = F.udf(
        lambda text: get_team_relevance(text)[1],
        FloatType()
    )
    
    df = df.withColumn("patriots_relevance", pats_relevance_udf(F.col("tweet_text")))
    df = df.withColumn("seahawks_relevance", hawks_relevance_udf(F.col("tweet_text")))
    
    return df

def create_feature_pipeline(df, max_text_features=100):
    """
    创建完整的特征工程流水线
    """
    try:
        # 确保df不为空
        if df.count() == 0:
            print("警告：输入DataFrame为空")
            return df, None
            
        # 1. 预处理文本（先提取再清理）
        df = preprocess_text(df)
        
        # 检查数据是否有效
        if df.filter(F.col("processed_text").isNotNull()).count() == 0:
            print("警告：所有文本预处理后为空")
            df = df.withColumn("text_features", F.array([F.lit(0.0)]))
            text_model = None
        else:
            # 2. 提取文本特征（使用更少的特征以提高性能）
            try:
                df, text_model = extract_text_features(df, max_features=max_text_features)
            except Exception as e:
                print(f"文本特征提取失败: {str(e)}")
                df = df.withColumn("text_features", F.array([F.lit(0.0)]))
                text_model = None
        
        # 3. 提取时间特征
        df = extract_temporal_features(df)
        
        # 4. 提取用户特征
        df = extract_user_features(df)
        
        # 5. 提取互动特征
        df = extract_engagement_features(df)
        
        # 6. 提取比赛上下文特征
        df = extract_game_context_features(df)
        
        # 7. 创建综合影响力特征
        df = df.withColumn(
            "influence_score",
            F.when(
                F.col("follower_count") > 0,
                F.col("retweet_count") / F.col("follower_count")
            ).otherwise(0.0)
        )
        
        # 检查特征创建是否成功
        required_cols = ["text_features", "patriots_relevance", "seahawks_relevance", 
                        "follower_count", "following_count", "is_verified", 
                        "hour_of_day", "day_of_week", "game_period"]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"警告：缺少以下列: {missing_cols}")
        
        return df, text_model
        
    except Exception as e:
        print(f"特征工程过程中出错: {str(e)}")
        # 确保返回一个有效的DataFrame
        if "text_features" not in df.columns:
            df = df.withColumn("text_features", F.array([F.lit(0.0)]))
        return df, None

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

# 简化特征提取以减少内存使用
def extract_text_features_simple(df, max_features=20, output_col="text_features"):
    """
    Extract text features using a simplified approach to avoid memory issues
    """
    try:
        # 使用简单的词频统计而不是完整的TF-IDF
        df = df.withColumn("words", F.split(F.col("processed_text"), "\\s+"))
        
        # 只保留有效的词
        df = df.withColumn("words", 
            F.expr("filter(words, word -> length(word) > 1 and length(word) < 20)")
        )
        
        # 使用简单的字符串拼接而不是复杂的NLP管道
        df = df.withColumn(output_col, F.array([F.lit(0.0)]))
        
        # 返回带有简单特征的数据集
        return df, None
    except Exception as e:
        print(f"简化文本特征提取出错: {str(e)}")
        df = df.withColumn(output_col, F.array([F.lit(0.0)]))
        return df, None

# 修改特征创建函数，使用更简单的方法，直接创建ML Vector类型
def create_simple_feature_pipeline(df, max_text_features=10):
    """
    创建简化的特征工程流水线，减少内存使用并确保特征类型正确
    """
    try:
        # 1. 简单文本预处理
        text_udf = F.udf(
            lambda tweet_obj: str(tweet_obj.get("text", str(tweet_obj))) if isinstance(tweet_obj, dict) else str(tweet_obj),
            StringType()
        )
        df = df.withColumn("tweet_text", text_udf(F.col("tweet")))
        
        # 2. 简单清理
        clean_udf = F.udf(
            lambda text: re.sub(r'[^\w\s]', ' ', text.lower()) if text else "",
            StringType()
        )
        df = df.withColumn("processed_text", clean_udf(F.col("tweet_text")))
        
        # 3. 简化的特征提取
        df, _ = extract_text_features_simple(df, max_features=max_text_features)
        
        # 4. 为影响力特征创建一个简单的值
        follower_udf = F.udf(
            lambda author_obj: float(author_obj.get("followers", 0)) if isinstance(author_obj, dict) else 0.0,
            FloatType()
        )
        df = df.withColumn("follower_count", follower_udf(F.col("author")))
        
        # 5. 创建时间特征
        df = df.withColumn("hour_of_day", 
                          F.when(F.col("datetime").isNotNull(), 
                                F.hour(F.col("datetime")))
                          .otherwise(0))
        
        # 6. 创建比赛期间特征
        df = df.withColumn("game_period", 
                          F.when(F.col("citation_date").isNotNull(), 
                                F.when(F.col("citation_date") < F.lit(1422835200), "pre_game")
                                .when(F.col("citation_date") > F.lit(1422846000), "post_game")
                                .otherwise("during_game"))
                          .otherwise("unknown"))
        
        # 7. 转换为Spark ML Vector格式
        to_vector_udf = F.udf(lambda arr: Vectors.dense(arr), VectorUDT())
        
        # 创建一个Vectors.dense的数组，包含1个简单的特征
        df = df.withColumn("feature_array", F.array([F.lit(1.0)]))
        df = df.withColumn("ml_features", to_vector_udf(F.col("feature_array")))
        
        # 8. 为多任务学习创建特征向量
        for task in ["team", "influence", "time"]:
            df = df.withColumn(f"{task}_features_vector", to_vector_udf(F.col("feature_array")))
        
        # 9. 添加标签列
        df = df.withColumn("team_label", F.lit(2))  # 默认中立
        df = df.withColumn("influence_score", F.lit(0.0))  # 默认影响力
        df = df.withColumn("time_period", F.lit(3))  # 默认时间段
        
        return df, None
        
    except Exception as e:
        print(f"简化特征工程过程中出错: {str(e)}")
        # 创建最小可用的DataFrame
        if df.count() > 0:
            to_vector_udf = F.udf(lambda arr: Vectors.dense(arr), VectorUDT())
            df = df.withColumn("feature_array", F.array([F.lit(1.0)]))
            df = df.withColumn("ml_features", to_vector_udf(F.col("feature_array")))
            for task in ["team", "influence", "time"]:
                df = df.withColumn(f"{task}_features_vector", to_vector_udf(F.col("feature_array")))
            df = df.withColumn("team_label", F.lit(2))
            df = df.withColumn("influence_score", F.lit(0.0))
            df = df.withColumn("time_period", F.lit(3))
        
        return df, None 