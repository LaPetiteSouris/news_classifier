import sklearn.datasets as datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as NaiveBayes

count_vect = CountVectorizer()


def load_data_training():
    data = datasets.load_files('train_data')
    return data


def load_test_data():
    test_data = datasets.load_files('test_data')
    test_count = count_vect.transform(test_data.data)
    return test_count


def process_raw_data():
    training_count = count_vect.fit_transform(load_data_training().data)
    return training_count


def classifier():
    bayes = NaiveBayes()
    bayes.fit(process_raw_data(), load_data_training().target_names)
    return bayes


bayes_clf = classifier()
res = bayes_clf.predict(load_test_data())
print res
