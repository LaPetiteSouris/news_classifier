import os
from classifier import bayesian_naive as clf
import numpy
# crawling news
os.system('scrapy crawl article')
# Load classifier
classifier = clf.load_classifier()
with open("web_crawl/result.jl") as file:
    data_to_predict = clf.load_json_feature_file(file)
bag_of_words = clf.test_data_process(data_to_predict)
# prediction
result = classifier.predict(bag_of_words)
# display result

with open("web_crawl/result.jl") as file:
    clf.dipslay_prediction_result(file, numpy.where(result == 1)[0])
# clean up
os.system('rm web_crawl/*.jl')
