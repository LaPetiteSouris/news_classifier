from classifier import bayesian_naive as bayes
import os

os.chdir(".")


class Test_Classifier:

    def test_json_parsing(self):
        with open('data/pytest_data.jl') as file:
            data = bayes.load_json_feature_file(file)
        assert len(data) > 0

    def test_process_training_data(self):
        with open('data/pytest_data.jl') as file:
            data = bayes.load_json_feature_file(file)
        labels = [1] * len(data)
        bag_of_words, labels = bayes.training_data_process(data, labels)
        assert bag_of_words.shape[0] == len(labels)

    def test_load_classifier(self):
        bayes_classifier = bayes.load_classifier()
        assert bayes_classifier is not None

    def test_prediction(self):
        bayes_classifier = bayes.load_classifier()
        prediction = bayes_classifier.predict(
            bayes.test_data_process(["test"]))
        assert prediction in [0, 1]

    def test_load_training_data(self):
        data, label = bayes.load_training_data()
        assert len(data) == len(label)

    def test_dipslay_prediction_result(self, capfd):
        result = [0, 1]
        with open("data/pytest_data.jl") as file:
            bayes.dipslay_prediction_result(file, result)
            out, err = capfd.readouterr()
            str = "The United Nations tribunal"
            assert str in out
            str = "President Obama met with the President Mauricio Macri"
            assert str in out
