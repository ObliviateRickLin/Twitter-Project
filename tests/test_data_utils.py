import unittest
import os
import sys
import json
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import init_spark, convert_unix_to_datetime, get_retweet_count, get_followers_count

class TestDataUtils(unittest.TestCase):
    """Test cases for data_utils.py"""

    def setUp(self):
        """Set up test fixtures"""
        # Initialize a spark session for testing
        self.spark = init_spark("TestDataUtils")
        
        # Create a sample tweet JSON for testing
        self.sample_tweet = {
            "text": "This is a test tweet #SuperBowl",
            "citation_date": 1423008750,  # Wed Feb 04 2015 01:25:50 GMT+0000
            "author": {
                "followers": 1000,
                "following": 500,
                "verified": True
            },
            "metrics": {
                "citations": {
                    "total": 25,
                    "replies": 5,
                    "retweets": 15,
                    "likes": 5
                }
            }
        }

    def tearDown(self):
        """Tear down test fixtures"""
        # Stop the spark session
        self.spark.stop()

    def test_init_spark(self):
        """Test that Spark session is initialized correctly"""
        self.assertIsNotNone(self.spark)
        self.assertEqual(self.spark.conf.get("spark.app.name"), "TestDataUtils")

    def test_convert_unix_to_datetime(self):
        """Test conversion of UNIX timestamp to datetime"""
        unix_time = 1423008750  # Wed Feb 04 2015 01:25:50 GMT+0000
        dt = convert_unix_to_datetime(unix_time)
        
        # Check that the conversion is correct
        self.assertIsInstance(dt, datetime)
        self.assertEqual(dt.year, 2015)
        self.assertEqual(dt.month, 2)
        self.assertEqual(dt.day, 4)
        
        # Test with timezone
        dt_pst = convert_unix_to_datetime(unix_time, timezone="America/Los_Angeles")
        self.assertEqual(dt_pst.hour, 17)  # 17:25:50 PST on Feb 3

    def test_get_retweet_count(self):
        """Test extraction of retweet count"""
        retweet_count = get_retweet_count(self.sample_tweet)
        self.assertEqual(retweet_count, 25)
        
        # Test with missing data
        empty_tweet = {}
        self.assertEqual(get_retweet_count(empty_tweet), 0)

    def test_get_followers_count(self):
        """Test extraction of followers count"""
        followers_count = get_followers_count(self.sample_tweet)
        self.assertEqual(followers_count, 1000)
        
        # Test with missing data
        empty_tweet = {}
        self.assertEqual(get_followers_count(empty_tweet), 0)

if __name__ == '__main__':
    unittest.main() 