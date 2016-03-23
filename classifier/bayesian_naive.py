from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB as NaiveBayes
import json
import pickle


count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()


def load_json_feature_file(web_json_file):
    """ This function load the JSONline file and parse
    all the text to a list, each member content the text of
    an article
    Args:
        Jsonfile: path to jsonelines file
    Return:
        data: list, where each member contains pure text of the article
    """
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
    """ Process from raw data to bag-of-words data, for reference
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    Args:
         raw_feature_vector: a list where each element is pure text document
         training_label: label class of the training data
    Return:
          bag_of_words: bag of word training data
          training_label: buffer traning label for each sample
    """
    training_count = count_vect.fit_transform(raw_feature_vector)
    bag_of_words_training = tfidf_transformer.fit_transform(training_count)
    return bag_of_words_training, training_label


def process_test_data(raw_feature_test_vector):
    """ Process from raw test data to bag-of-words data, for reference
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    Args:
         raw_feature_vector: a list where each element is pure text document
    Return:
          bag_of_words: bag of word training data
    """
    test_count = count_vect.transform(raw_feature_test_vector)
    bag_of_word_test = tfidf_transformer.transform(test_count)
    return bag_of_word_test


def save_classifier(classifier):
    """ Save classifier object to pickle
    Arg:
        classifier: classifier to be saved
    Return:
        None
    """
    f = open('bayesian_classifier.pickle', 'wb')
    pickle.dump(classifier, f, -1)
    f.close()


def load_classifier():
    """ Load classifier object from pickle. If no pickle found, it will
    create an empty classifier object
    Args:
         None
    Return:
          classifier: trained-classifier object
    """
    try:
        f = open('bayesian_classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
    except IOError:
        print 'Bayesian classifier object not found. Will create now'
        classifier = NaiveBayes()
        bag_of_word, label = process_training_data(
            ["Positive", "Negative"], [1, 0])
        classifier.partial_fit(bag_of_word, label, label)
    return classifier
