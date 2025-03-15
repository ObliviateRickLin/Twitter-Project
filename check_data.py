import json
import os

# 确保输出目录存在
os.makedirs("analysis", exist_ok=True)

# 打开输出文件
with open("analysis/data_sample.txt", "w", encoding="utf-8") as outfile:
    for hashtag in ["SuperBowl", "NFL", "patriots", "gopatriots", "gohawks", "sb49"]:
        outfile.write(f"\nChecking #{hashtag} data:\n")
        try:
            with open(f"ECE219_tweet_data/tweets_#{hashtag}.txt", "r", encoding="utf-8") as f:
                # 读取第一条推文
                line = f.readline().strip()
                tweet = json.loads(line)
                outfile.write("\nAvailable fields: " + str(list(tweet.keys())) + "\n")
                outfile.write("\nSample values:\n")
                tweet_text = tweet.get("tweet", "N/A")
                if isinstance(tweet_text, str):
                    tweet_text = tweet_text[:100] + "..." if len(tweet_text) > 100 else tweet_text
                outfile.write("- tweet: " + str(tweet_text) + "\n")
                outfile.write("- author: " + str({k: v for k, v in tweet.get("author", {}).items() if k in ["followers", "following", "verified"]}) + "\n")
                outfile.write("- metrics: " + str(tweet.get("metrics", {}).get("citations", {})) + "\n")
                outfile.write("- firstpost_date: " + str(tweet.get("firstpost_date")) + "\n")
                outfile.write("- citation_date: " + str(tweet.get("citation_date")) + "\n")
                outfile.write("\n" + "="*50 + "\n")
        except FileNotFoundError:
            outfile.write(f"File not found for #{hashtag}\n")
        except json.JSONDecodeError:
            outfile.write(f"Invalid JSON in #{hashtag} file\n")
        except Exception as e:
            outfile.write(f"Error processing #{hashtag}: {str(e)}\n")

print("数据样本已保存到 analysis/data_sample.txt") 