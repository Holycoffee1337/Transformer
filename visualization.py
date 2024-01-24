'''
File: visualization.py
Purpose: Provide functions for visualization, 
of the give data.
'''
import pandas as pd
from pandas._libs.hashtable import value_count
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import Embedding, LSTM, Dense, Dropout, LayerNormalization
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
# Own Files imported
import data_preprocessing 

def common_words(df_format):
    """
    Analyzes the text data in the 'text' column of the provided DataFrame and
    prints the top 10 most common words along with their frequencies.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'text' column to analyze.

    Returns:
        None

    Execution: 
        Show plot
    """
    
    path = 'common_words' 




    # Handle NaN values by replacing them with an empty string
    df_format['Text'] = df_format['Text'].replace(np.nan, '', regex=True)
    # Use CountVectorizer to tokenize and count word frequencies
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_format['Text'])
    
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum the occurrences of each word across all sentences
    word_frequencies = Counter(dict(zip(feature_names, X.sum(axis=0).A1)))
    # print(word_frequencies['positive'])
    # Remove specific keys
    # word_frequencies.pop('positive', None)
    # word_frequencies.pop('negative', None)
    
    # Display the most common words and their frequencies
    most_common_words = word_frequencies.most_common(10)
    most_common_words = most_common_words[0:10]
    #for word, frequency in most_common_words:
        # print(f"{word}: {frequency}")
    # Plot a bar chart of word frequencies
    plt.figure(figsize=(12, 8))
    plt.bar(*zip(*most_common_words))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Words (With stopwords)')
    plt.savefig(path)
    plt.close()
    plt.show()


def count_words(text):
    if pd.isna(text):  # Check for NaN values
        return 0
    words = str(text).split()  # Convert to string and split
    return len(words)


def word_count_distribution(df):


    path = 'word_count_distribution'
    df['Word_Count'] = df['Text'].apply(count_words)
    # print(df['Word_Count'].mean())

    percentiles = df['Word_Count'].describe(percentiles=[0.25, 0.5, 0.75])

    word_count_dict = {}
    for count in df['Word_Count']:
        if count in word_count_dict:
            word_count_dict[count] += 1
        else:
            word_count_dict[count] = 1
    
    # Convert the dictionary items to lists for plotting
    word_counts = list(word_count_dict.keys())
    row_counts = list(word_count_dict.values())

    # Create fig
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12, 8))

    table_data = [['25%', f"{percentiles['25%']:.2f}"],
                ['50%', f"{percentiles['50%']:.2f}"],
                ['75%', f"{percentiles['75%']:.2f}"]]
    table = ax2.table(cellText=table_data, loc='center', 
                     colLabels=['Percentile', 'Value'], cellLoc='center', colColours=['#f0f0f0']*2)


    ax2.axis('off')
    # Plotting the bar chart
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Number of Rows')
    ax1.set_title('Word Count Distribution in Rows')
    ax1.bar(word_counts, row_counts, color='blue')
    # plt.show()
    plt.savefig(path)
    plt.close()
 




