# Imports
import pandas as pd

# load the data
df = pd.read_csv('../data/creditcard_topics.csv')

# Drop these topics: Bill Disputes and Fraud Issues
df = df[(df.topics != 'Bill Disputes') & (df.topics != 'Fraud Issues')]

# Review count
review_count = df.groupby('creditcards')[
    'sentences'].count().sort_values(ascending=False)
print(review_count.head())


def webapp_data(data, threshold=None):
    """This function filters all the results in a dataframe to be used in a web app"""
    df = pd.DataFrame()  # Initialize dataframe
    cards = list(data.creditcards.unique())  # List of all credit cards
    for card in cards:
        data = data[['topics', 'creditcards',
                     'sentences', 'sentiments', 'mean_sentiments']]
        df_sent = data[data.creditcards ==
                       card]['sentences'].reset_index(drop=True)
        sent = df_sent.loc[0] + 'varsep' + df_sent.loc[1] + 'varsep' + df_sent.loc[2] +\
            'varsep' + df_sent.loc[len(df_sent) - 1] + 'varsep' + df_sent.loc[len(df_sent) - 2] +\
            'varsep' + \
            df_sent.loc[len(
                df_sent) - 3]  # 3 positive and 3 negative sentences for each card
        # positive topics for sentiment greater than threshold
        pos_topic = data[data.sentiments > threshold]
        # negative topics for sentiment greater than threshold
        neg_topic = data[data.sentiments < threshold]
        pos_topic = pos_topic[pos_topic.creditcards == card].reset_index(
            drop=True)  # positives for each card
        neg_topic = neg_topic[neg_topic.creditcards == card].reset_index(
            drop=True)  # negatives for each card
        # No of positive sentences under a topic
        pos_dict = dict(pos_topic.groupby('topics')['sentiments'].count())
        # No of negative sentences under a topic
        neg_dict = dict(neg_topic.groupby('topics')['sentiments'].count())
        # dictionary to hold the number of positive and negative sentences under a topic
        total_dict = dict()
        for pos, neg in zip(pos_dict.items(), neg_dict.items()):
            total_dict[pos[0]] = [pos[1], neg[1]]
        # for google bar chart display
        task_dict = {'Task': ['Satisfied', 'Unsatisfied']}
        # join dictionary so that  task_dict appears first
        topic_dict = {**task_dict, **total_dict}
        mean_sentiment = data[data.creditcards ==
                              card]['mean_sentiments'].reset_index(drop=True).loc[0]
        df = df.append({'creditcard': card, 'topic': topic_dict,
                        'avg_sentiment': mean_sentiment,
                        'pos_neg_sentence': sent}, ignore_index=True)
    return df


# Extract the data from the function
data = webapp_data(df, threshold=0.1)
# print data
print(data.head())

# Add total number of review for each credit card
data['num_reviews'] = data.creditcard
data['num_reviews'] = data['num_reviews'].map(dict(review_count))

# Save data as csv file
data.to_csv('../data/creditcard_data.csv', index=False)
