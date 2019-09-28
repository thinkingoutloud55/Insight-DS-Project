# Created by Solomon Owerre.
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from collections import Counter


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
