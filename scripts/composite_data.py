~# Imports
import pandas as pd

# load the data
df = pd.read_csv('../data/creditcard_topics.csv')

# Drop these topics: Bill Disputes and Fraud Issues
df = df[(df.topics != 'Bill Disputes') & (df.topics != 'Fraud Issues')]


def filter_data(data, threshold=None):
    """ Returns a dataframe of all the credit cards to be used in the web app"""
    df = pd.DataFrame()  # Initialize dataframe
    cards = list(data.creditcards.unique())  # List of all credit cards
    for card in cards:
        data = data[['topics', 'creditcards',
                     'sentences', 'sentiments', 'mean_sentiments']]
        df_sent = data[data.creditcards ==
                       card]['sentences'].reset_index(drop=True)
        sent = df_sent.loc[0] + 'varsep' + df_sent.loc[1] + 'varsep' + df_sent.loc[2] +\
            'varsep' + df_sent.loc[len(df_sent)-1] + 'varsep' + df_sent.loc[len(df_sent)-2] +\
            'varsep' + \
            df_sent.loc[len(
                df_sent)-3]  # 3 positive and 3 negative sentences for each card
        # positive topics for sentiment greater than threshold
        df_topic = data[data.sentiments > threshold]
        df_topic = df_topic[df_topic.creditcards ==
                            card].reset_index(drop=True)
        task_dict = {'Task': 'Topic distribution'}  # for google charts
        # No of positive sentences under a topic
        group_dict = dict(df_topic.groupby('topics')['sentiments'].count())
        # join dictionary so that  task_dict appears first
        topic_dict = {**task_dict, **group_dict}
        mean_sentiment = data[data.creditcards ==
                              card]['mean_sentiments'].reset_index(drop=True).loc[0]
        df = df.append({'creditcard': card, 'topic': topic_dict,
                        'avg_sentiment': mean_sentiment, 'pos_neg_sentence': sent},
                       ignore_index=True)
    return df


# Extract the data from the function
data = filter_data(df, threshold=0.1)


# Total number of review sentences for each credit card
review_num = {'Chase Amazon Reward Visa': 10398,
              'Credit One Bank': 9565,
              'Bank of America Cash Rewards Credit Card': 6177,
              'Capital One Quicksilver Rewards': 5532,
              'Merrick Bank': 4913,
              'Capital One Venture Rewards': 4669,
              'Capital One Platinum': 4552,
              'Chase Freedom Unlimited': 3250,
              'Target Credit Card': 2669,
              'American Express Platinum Card': 2578,
              'Citi Double Cash Card': 2525,
              'Discover it Cash Back': 2418,
              'Citi Simplicity Card': 2384,
              'Citi Diamond Preferred Card': 2109,
              'Bank of America Travel Rewards Credit Card': 2048,
              'Premier Bankcard': 2022,
              'PayPal Credit': 2000,
              'Eppicard': 1860,
              'Capital One Secured Credit Card': 1756,
              'American Express Blue Cash Preferred': 1723,
              'Barclays Bank': 1630,
              'American Express Business Gold Rewards Credit Card': 1215,
              'Chase Sapphire Preffered Card': 1100,
              'OpenSky Secured Visa Credit Card': 916,
              'Sears Credit Card': 856,
              'TD Cash Visa Credit Card': 735,
              'Capital One Platinum Costco MASTERCARD': 537,
              'Discover it Secured': 377,
              'Presidents Choice Financial': 358,
              'Discover it Miles': 337,
              'Walmart MASTERCARD': 167,
              'Rogers MASTERCARD': 163,
              'Aspire Visa': 145,
              'Canadian Tire MASTERCARD': 141,
              'Costco Anywhere Visa Card by Citi': 108,
              'TD First Class Travel VISA Infinite Card': 99,
              'HomeTrust Secured VISA': 93,
              'RBC Avion Visa Infinite': 52,
              'BMO MASTERCARD': 38,
              'Citi Platinum World Elite': 32}

# Add total number of review for each credit card
data['num_reviews'] = data.creditcard
data['num_reviews'] = data['num_reviews'].map(review_num)

# Save data as csv file
data.to_csv('../data/creditcard_data.csv', index=False)
