# news_classifier

A simple news recommendation system.<br/> 

It parses International news headlines from New York Times and filter news those are classified as not likely to be read by user <br/>

A training data set is stored in a zip file in *data/data.zip*. To update training data set, please update the files here <br/>
#Required <br/>
virtualenv 
<br/>

# Training data set <br/>

Should be stored in *data/data.zip*. There are 2 files <br/>

1.pos.jl : Each lines contains a json object of an article **likely to** be read <br/>
2.neg.jl : Each lines contains a json object of an article **unlikely** to be read <br/>

# How to set up <br/>

1. Create and launch  your virtual evn <br/>
2. In your virtual env, run   *pip install requirements.txt*  to set up dependencies correctly
3. Launch *python launch_crawling.py* from your env to get news feeds
# Pytest <br/>

In your env terminal, launch *test.py -v py.test*
