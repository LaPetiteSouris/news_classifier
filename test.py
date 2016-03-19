from classifier import data_crawling as crawler


class Test_Classifier:

    def test_data_loading(self):
        bag_of_words_training, training_label = crawler.load_training_data()
        assert bag_of_words_training.shape[0] == len(training_label)

    def test_bayesian_classifier(self):
        bayes = crawler.classifier()
        assert bayes.predict(crawler.load_test_data()) == 'animals'
