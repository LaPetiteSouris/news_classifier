from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB as NaiveBayes
import json
import pickle


count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()


def load_json_feature_file(web_json_file):
    data = []
    with open(web_json_file) as f:
        for line in f:
            parsed_json = json.loads(line)
            title = ''.join(''.join(element)
                            for element in parsed_json['title'])
            content = ''.join(''.join(element)
                              for element in parsed_json['content'])
            text = ''.join([title, content])
            data.append(text)
    return data


def process_training_data(raw_feature_vector, training_label):
    training_count = count_vect.fit_transform(raw_feature_vector)
    bag_of_words_training = tfidf_transformer.fit_transform(training_count)
    return bag_of_words_training, training_label


def process_test_data(raw_feature_test_vector):
    test_count = count_vect.transform(raw_feature_test_vector)
    bag_of_word_test = tfidf_transformer.transform(test_count)
    return bag_of_word_test


def save_classifier(classifier):
    f = open('bayesian_classifier.pickle', 'wb')
    pickle.dump(classifier, f, -1)
    f.close()


def load_classifier():
    try:
        f = open('bayesian_classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
    except IOError:
        print 'Bayesian classifier object not found. Will create now'
        classifier = NaiveBayes()
    return classifier
