from classifier import bayesian_naive as bayes


class Test_Classifier:

    def test_json_parsing(self):
        data = bayes.load_json_feature_file('web_crawl/result.jl')
        assert len(data) > 0

    def test_process_training_data(self):
        data = bayes.load_json_feature_file('web_crawl/result.jl')
        labels = [1] * len(data)
        bag_of_words, labels = bayes.process_training_data(data, labels)
        assert bag_of_words.shape[0] == len(labels)

    def test_load_classifier(self):
    	bayes_classifier=bayes.load_classifier()
