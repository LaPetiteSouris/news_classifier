import os
from classifier import bayesian_naive as clf


os.system('scrapy crawl article')
classifier = clf.load_classifier()
with open("web_crawl/result.jl") as file:
    test_data = clf.load_json_feature_file(file)
bag_of_words = clf.test_data_process(test_data)
result = classifier.predict(bag_of_words)
print result
