# Created by Solomon Owerre.
import re
import nltk
import gensim
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from collections import Counter
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel, LdaMulticore


def review_to_sent(data, review, company):
    """Converts reviews to sentences and keeps track of company name for each sentence"""
    sentence_tokens = [[sentence for sentence in sent_tokenize(
        data[review].loc[i])] for i in range(len(data))]
    count_sentences = [len(x) for x in sentence_tokens]
    sentences = [
        sentence for sub_sentence in sentence_tokens for sentence in sub_sentence]
    count_company = [[x] for x in data[company]]
    company_token = []
    for idx, val in enumerate(count_sentences):
        company_token.append(count_company[idx] * val)
    company_names = [name for names in company_token for name in names]
    return sentences, company_names


def print_review(data, index=None):
    """Display index row of the review dataframe"""
    text = data[data.index == index].values.reshape(3)
    review = text[0]
    rating = text[1]
    company = text[2]
    print('Review:', review)
    print('Rating:', rating)
    print('Credit card:', company)


def print_sentence(data, index=None):
    """Display index row of the sentence dataframe"""
    text = data[data.index == index].values.reshape(4)
    sentence = text[0]
    company = text[1]
    sentiment = text[2]
    positivity = text[3]
    print('Sentence:', sentence)
    print('Credit card:', company)
    print('Sentiment:', sentiment)
    print('Positivity:', positivity)


def pre_process_text(data, text_string):
    """clean the review texts"""
    data[text_string] = data[text_string].str.replace(r"http\S+", "")
    data[text_string] = data[text_string].str.replace(r"http", "")
    data[text_string] = data[text_string].str.replace(r"@\S+", "")
    data[text_string] = data[text_string].str.replace(
        r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    data[text_string] = data[text_string].str.replace(r"@", "at")
    data[text_string] = data[text_string].str.replace(r"\n", "")
    data[text_string] = data[text_string].str.replace(r"\t", "")
    data[text_string] = data[text_string].str.replace(r"\d+", "")
    data[text_string] = data[text_string].str.lower()
    return data


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def word_lemmatizer(text):
    """Word lemmatization function"""
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text, get_wordnet_pos(text))
    return text


# By inspection, I will add more stopwords to the default nltk stopwords
my_stop_words = ['capital', 'america', 'redcard', 'target', 'amazon', 'card', 'credit', 'merrick', 'discover', 'citi',
                 'amex', 'express', 'go', 'paypal', 'chase', 'american', 'one', 'would', 'ask', 'really',
                 'get', 'know', 'express', 'ever', 'use', 'say', 'recently', 'also', 'always', 'give',  'tell',
                 'take', 'never', 'costco', 'time', 'make', 'try', 'number', 'send', 'new', 'even',
                 'sony', 'us', 'husband', 'car', 'capitol', 'wife', 'book', 'could', 'okay', 'mastercard', 'want',
                 'honestly', 'eppicard', 'need', 'family', 'cap', 'another', 'line', 'com', 'fico', 'quicksilver',
                 'link', 'sear', 'pay', 'may', 'company', 'bank', 'call', 'account', 'receive', 'told', 'day', 'well',
                 'think', 'look', 'sure', 'easy', 'money', 'people', 'business', 'review', 'something', 'come', 'away']

stop_words = stopwords.words('english')
stop_words.extend(my_stop_words)


def my_tokenizer(text):
    """Tokenize review text"""
    text = text.lower()  # lower case
    text = re.sub("\d+", " ", text)  # remove digits
    tokenizer = RegexpTokenizer(r'\w+')
    token = [word for word in tokenizer.tokenize(text) if len(word) > 2]
    token = [word_lemmatizer(x) for x in token]
    token = [s for s in token if s not in stop_words]
    return token


def detokenizer(text):
    """Remove wide space in review texts"""
    my_detokenizer = TreebankWordDetokenizer().detokenize(sent_tokenize(text))
    return my_detokenizer


def topic_threshold(doc_topic, topic_vector, threshold=None):
    """Return the topic number if the topic is more than 70% dominant"""
    topic_num_list = []
    for i in range(len(topic_vector)):
        topic_num = [idx for idx, value in enumerate(
            doc_topic[i]) if value > threshold]
        if topic_num != []:
            topic_num = topic_num[0]
        else:
            topic_num = 'None'
        topic_num_list.append(topic_num)
    return topic_num_list


def compute_coherence_lda(corpus, dictionary, start=None, limit=None, step=None):
    """Compute c_v coherence for various number of topics """
    topic_coherence = []
    model_list = []
    tokens_list = df.trigram_tokens.values.tolist()
    texts = [[token for sub_token in tokens_list for token in sub_token]]
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, random_state=0, num_topics=num_topics,
                         alpha='auto', eta='auto')
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        topic_coherence.append(coherencemodel.get_coherence())

    return model_list, topic_coherence


def add_bigram(token_list):
    """add bigrams in the data"""
    bigram = gensim.models.Phrases(token_list)
    bigram = [bigram[line] for line in token_list]
    return bigram


def add_trigram(token_list):
    """add trigrams in the data"""
    bigram = add_bigram(token_list)
    trigram = gensim.models.Phrases(bigram)
    trigram = [trigram[line] for line in bigram]
    return trigram


def doc_term_matrix(data, text):
    """Returns document-term matrix, vocabulary, and word_id"""
    counter = CountVectorizer(tokenizer=my_tokenizer, ngram_range=(1, 1))
    data_vectorized = counter.fit_transform(data[text])
    X = data_vectorized.toarray()
    bow_docs = pd.DataFrame(X, columns=counter.get_feature_names())
    vocab = tuple(bow_docs.columns)
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    return data_vectorized, vocab, word2id, counter


def credit_card(data):
    """Returns topics, topic_sentiments, and credit cards"""
    topic_list = data.topics.unique()
    card_lists = []
    topic_sentiment = []
    topics = []
    for i in range(len(topic_list)):
        specific = df[df.topics == topic_list[i]][[
            'topics', 'creditcards', 'sentiments']].reset_index(drop=True)
        group_table = specific.groupby('creditcards')[
            'sentiments'].mean().sort_values(ascending=False)
        card_lists.append(list(group_table.index))
        topic_sentiment.append(group_table.values.round(2))
        for j in range(len(group_table)):
            topics.append(topic_list[i])

    # Convert the list of lists to list
    card_list = [card for sub_card in card_lists for card in sub_card]
    topic_sen = [sen for sub_sen in topic_sentiment for sen in sub_sen]

    return topics, card_list, topic_sen
