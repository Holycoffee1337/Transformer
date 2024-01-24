import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# TICKER = 'TSLA'
# TICKER = 'RHHBY'
# TICKER = 'MRNA' 
# TICKER = 'NVDA'
# TICKER = 'META'
# TICKER = 'LVMUY'

TICKER = 'JPM'
START = '2023-12-01'
END = '2024-01-22'
print('Executing')


# Creating and saving timeseries .csv
def timeseries_to_csv(ticker, START, END):
    path = './Data/timeseries/'
    stock = yf.download(ticker, start=START, end=END)
    stock.to_csv(path + ticker + '.csv', index=True)

def get_timeseries(ticker):
    path = './Data/timeseries/'
    df = pd.read_csv(path + ticker + '.csv')
    return df


# Get articles, for specific company
def get_articles(company_name, from_date, to_date):

    api_key = 'df848821e45a4ae99876dfe0ad12399a'
    all_articles = []
    page = 1
    total_pages = 1  # Initialize with 1 to start the while loop

    while page <= total_pages:
        query = f'{company_name} stock news'
        url = f'https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}&page={page}'

        response = requests.get(url)
        data = response.json()

        if 'articles' in data:
            all_articles.extend(data['articles'])
            
            # Check if totalResults is present and calculate total pages
            if 'totalResults' in data:
                total_results = data['totalResults']
                total_pages = (total_results // 100) + (total_results % 100 > 0)
            
            page += 1
        else:
            break

    return all_articles


# Returning specific articles for companies 
def getArticle(company_name, from_date, to_date):

    api_key = 'df848821e45a4ae99876dfe0ad12399a'
    
    query = f'{company_name} stock news'
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    
    response = requests.get(url)
    data = response.json()
    
    # Check if 'articles' key is present in the response
    if 'articles' in data:
        articles = data['articles']
        num_articles = len(articles)
        
        # Create a DataFrame from the articles
        df = pd.DataFrame(articles)
        
        # Rename the columns to match the desired output
        df = df[['title', 'publishedAt']].copy()
        df.rename(columns={'title': 'Headline', 'publishedAt': 'Date'}, inplace=True)

        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df.sort_values(by='Date', inplace=True)

        print(f'Total number of articles: {num_articles}')

        for i, article in enumerate(articles[:5]): 
            headline = article['title']
            date = article['publishedAt']
    else:
                print('No articles found in the response.')
    return df 
    

# Plot the timeseries for given ticker 
def plot_price(df, ticker):

    plt.figure(figsize=(10, 3))
    plt.title(ticker + ": Adj. Close price")
    plt.xlabel("$t$")

    # Adding table
    price = df['Adj Close']
    price = price.reset_index(drop=True)
    date_df = df['Date']
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)


    plt.plot(df.index, price, label="Price", color="blue")
    plt.legend()
    plt.show()


# Plot timeseries price with article labels
def plot_price_with_labels(time_series_df, articles_df, ticker):
    # Convert 'Date' to datetime in both DataFrames and set it as index
    time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
    time_series_df.set_index('Date', inplace=True)
    
    articles_df['Date'] = pd.to_datetime(articles_df['Date'])
    articles_df.set_index('Date', inplace=True)

    # Merge the two DataFrames on the 'Date' index
    merged_df = time_series_df.merge(articles_df, how='left', left_index=True, right_index=True)

    # Plotting the time series
    plt.figure(figsize=(10, 5))
    plt.title(ticker + ": Adj. Close Price with Article Sentiment")
    plt.xlabel("Date")
    plt.ylabel("Adj. Close Price ($)")

    # Plot the Adj. Close price
    plt.plot(merged_df.index, merged_df['Adj Close'], label="Price", color="blue")

    # Add dots for article sentiment
    for date, row in merged_df.iterrows():
        if pd.notna(row['Label']):  # Check if there's a label for that date
            color = 'green' if row['Label'] == 'positive' else 'red'
            plt.plot(date, row['Adj Close'], 'o', color=color)  # Plot the dot

    plt.legend()
    plt.show()



def plot_price_with_majority_vote(time_series_df, articles_df, ticker):

    """
    Plot stock timeseries, with labels (and majority votes)
    """

    time_series_df['Date'] = pd.to_datetime(time_series_df['Date'])
    time_series_df.set_index('Date', inplace=True)
    
    articles_df['Date'] = pd.to_datetime(articles_df['Date'])

    # Majority Vote 
    majority_vote = articles_df.groupby('Date')['Label'].agg(
        lambda x: 'neutral' if x.value_counts().iloc[0] == x.value_counts().iloc[-1] else x.value_counts().idxmax()
    )
    
    # Merge the time series data with the majority vote labels
    merged_df = time_series_df.merge(majority_vote, how='left', left_index=True, right_index=True)

    # Plotting the time series
    plt.figure(figsize=(10, 5))
    plt.title(ticker + ": Adj. Close Price with Article Sentiment")
    plt.xlabel("Date")
    plt.ylabel("Adj. Close Price ($)")
    plt.plot(label='Positive articles', color='green')

    # Plot the Adj. Close price
    plt.plot(merged_df.index, merged_df['Adj Close'], label="Price", color="blue")

    for date, row in merged_df.iterrows():
        if pd.notna(row['Label']):  # Check if there's a label (majority vote or neutral) for that date
            color = 'green' if row['Label'] == 'positive' else ('red' if row['Label'] == 'negative' else 'grey')
            plt.plot(date, row['Adj Close'], 'o', color=color)  # Plot the dot

    plt.scatter([], [], color='green', label='Positive Sentiment')
    plt.scatter([], [], color='red', label='Negative Sentiment')
    plt.scatter([], [], color='grey', label='Neutral Sentiment')
    plt.legend()
    plt.savefig(f'{TICKER}_plot.png')
    plt.show()
    plt.close()


if __name__ == "__main__":

    """
    To create and save the plots
    """


    articles_df = pd.read_csv(f"{TICKER}_classified_articles.csv")
    timeseries_to_csv(TICKER, START, END)
    stock_timeseries = get_timeseries(TICKER)
    print(stock_timeseries)
    plot_price_with_majority_vote(stock_timeseries, articles_df, TICKER)
