# Imports
import pandas as pd
import numpy as np

# load the data
df = pd.read_csv('../data/creditcard_topics.csv')

# Drop these topics: Bill Disputes and Fraud Issues
df = df[(df.topics != 'Bill Disputes')&(df.topics != 'Fraud Issues')]

def filter_data(data, threshold = None):
    """ Returns a dataframe of all the credit cards to be used in the web app"""
    df = pd.DataFrame() # Initialize dataframe
    cards = list(data.creditcards.unique()) # List of all credit cards
    for card in cards:
        data = data[['topics', 'creditcards', 'sentences', 'sentiments', 'mean_sentiments']]
        df_sent = data[data.creditcards== card]['sentences'].reset_index(drop = True)
        sent = df_sent.loc[0] + 'varsep' + df_sent.loc[1] + 'varsep'+  df_sent.loc[2] +\
                'varsep'+ df_sent.loc[len(df_sent)-1] + 'varsep'+ df_sent.loc[len(df_sent)-2] +\
                'varsep'+ df_sent.loc[len(df_sent)-3] # 3 positive and 3 negative sentences for each card
        df_topic = data[data.sentiments > threshold] # positive topics for sentiment greater than threshold
        df_topic= df_topic[df_topic.creditcards == card].reset_index(drop= True)
        task_dict = {'Task': 'Topic distribution'} # for google charts
        group_dict = dict(df_topic.groupby('topics')['sentiments'].count())# No of positive sentences under a topic
        topic_dict = {**task_dict, **group_dict} # join dictionary so that  task_dict appears first
        mean_sentiment =data[data.creditcards== card]['mean_sentiments'].reset_index(drop= True).loc[0]
        df = df.append({'creditcard': card, 'topic': topic_dict,
                        'avg_sentiment':mean_sentiment, 'pos_neg_sentence':sent},
                        ignore_index = True)
    return df

# Extract the data from the function
data = filter_data(df, threshold=0.1)


# Total number of reviews for each credit card
review_num = {'Bank of America Cash Rewards Credit Card': 2476,
 'Capital One Quicksilver Rewards': 2292,
 'Capital One Venture Rewards': 2204,
 'Capital One Platinum': 2103,
 'Credit One Bank': 2016,
 'Chase Amazon Reward Visa': 1990,
 'Merrick Bank': 1578,
 'Citi Double Cash Card': 1004,
 'Citi Diamond Preferred Card': 900,
 'Bank of America Travel Rewards Credit Card': 898,
 'Citi Simplicity Card': 883,
 'Discover it Cash Back': 801,
 'American Express Blue Cash Preferred': 756,
 'Target Credit Card': 588,
 'American Express Platinum Card': 584,
 'Chase Freedom Unlimited': 551,
 'American Express Business Gold Rewards Credit Card': 462,
 'PayPal Credit': 437,
 'Eppicard': 425,
 'Premier Bankcard': 423,
 'Barclays Bank': 338,
 'TD Cash Visa Credit Card': 328,
 'Capital One Secured Credit Card': 292,
 'Chase Sapphire Preffered Card': 197,
 'Discover it Secured': 189,
 'Discover it Miles': 189,
 'Sears Credit Card': 186,
 'OpenSky Secured Visa Credit Card': 181,
 'Capital One Platinum Costco MASTERCARD': 90,
 'Presidents Choice Financial': 60,
 'Costco Anywhere Visa Card by Citi': 38,
 'Canadian Tire MASTERCARD': 37,
 'Rogers MASTERCARD': 36,
 'Walmart MASTERCARD': 32,
 'Aspire Visa': 31,
 'BMO MASTERCARD': 29,
 'TD First Class Travel VISA Infinite Card': 24,
 'HomeTrust Secured VISA': 20,
 'RBC Avion Visa Infinite': 20,
 'Citi Platinum World Elite': 20}

# Add total number of review for each credit card
data['num_reviews'] = data.creditcard
data['num_reviews'] = data['num_reviews'].map(review_num)

# Save data as csv file
data.to_csv('../data/creditcard_data.csv', index = False)
