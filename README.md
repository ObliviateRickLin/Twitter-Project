# Twitter Data Analysis Project

This project is based on machine learning analysis of Twitter data, completing the requirements of ECE219 Project 4. The project uses PySpark to process large-scale Twitter data, including basic statistical analysis, feature engineering, and machine learning model training.

## Project Structure

```
HW4/
├── ECE219_tweet_data/                # Directory of original Twitter data
│   ├── tweets_#superbowl.txt         # Super Bowl related tweets
│   ├── tweets_#sb49.txt              # Super Bowl 49 related tweets
│   ├── tweets_#patriots.txt          # Patriots related tweets
│   ├── tweets_#nfl.txt               # NFL related tweets
│   ├── tweets_#gopatriots.txt        # Tweets supporting the Patriots
│   └── tweets_#gohawks.txt           # Tweets supporting the Seahawks
│
├── analysis/                         # Analysis results and charts
│   ├── figures/                      # Save generated charts
│   ├── models/                       # Model results
│   └── stats/                        # Statistical data
│
├── tests/                            # Test directory
│   └── test_data_utils.py            # Data processing function test
│
├── data_utils.py                     # Data loading and processing tools
├── basic_stats.py                    # Calculate basic statistics
├── feature_engineering.py            # Feature engineering function
├── model.py                          # Model implementation
├── visualization.py                  # Visualization function
├── main.py                           # Main program
└── README.md                         # Project description
```

## Environment Requirements

- Python 3.7+
- PySpark 3.0+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- wordcloud

You can install the required dependencies with the following command:

```bash
pip install pyspark pandas numpy matplotlib seaborn scikit-learn wordcloud pytz
```

## Results

The analysis results will be saved in the `analysis` directory:
- `analysis/figures/`: Contains the generated charts
- `analysis/stats/`: Contains the basic statistical data
- `analysis/models/`: Contains the model performance results