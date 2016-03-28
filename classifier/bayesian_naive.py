from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB as NaiveBayes
import json
import pickle
import zipfile

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()


def load_json_feature_file(web_json_file):
    """ This function load the JSONline file and parse
    all the text to a list, each member content the text of
    an article.
    Args:
        Jsonfile: path to jsonelines file
    Return:
        data: list, where each member contains pure text of the article
    """
    data = []
    for line in web_json_file:
        parsed_json = json.loads(line)
        title = ''.join(''.join(element)
                        for element in parsed_json['title'])
        content = ''.join(''.join(element)
                          for element in parsed_json['content'])
        text = ''.join([title, content])
        data.append(text)
    return data


def dipslay_prediction_result(web_json_file, result_list):
    ''' This function display recommended news article to
    users
    Args:
        web_json_file: cralwed jsonlines document
        result_list: prediction list, contain index
        (line number in web_json_file)
        of positive article
    Returns:
        None
    '''
    for index, line in enumerate(web_json_file):
        if index in result_list:
            try:
                parsed_json = json.loads(line)
                title = parsed_json['title'][0]
                content = parsed_json['content'][0]
                url = parsed_json['link'][0]
                print "Title: %s\nContent: %s\nUrl: %s\n" % (title,
                                                             content, url)
            except IndexError:
                pass


def training_data_process(raw_feature_vector, training_label):
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


def test_data_process(raw_feature_test_vector):
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


def load_training_data():
    ''' This function load training data set, which located in a zip file
    Args: None
    Returns: None
    '''
    data = []
    label = []
    # Load positive result and negative result from zip file
    zip_f = zipfile.ZipFile('data/data.zip', 'r')
    file_pos = zip_f.open('pos.jl')
    file_neg = zip_f.open('neg.jl')
    data_pos = load_json_feature_file(file_pos)
    data_neg = load_json_feature_file(file_neg)
    label_pos = [1] * len(data_pos)
    label_neg = [0] * len(data_neg)
    # Construct a training data list and its label
    data.extend(data_pos)
    data.extend(data_neg)
    label.extend(label_pos)
    label.extend(label_neg)
    return data, label


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
        classifier = NaiveBayes(alpha=0.01)
        data, label = load_training_data()
        bag_of_word, label = training_data_process(
            data, label)
        classifier.fit(bag_of_word, label)
    return classifier
