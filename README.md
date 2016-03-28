# news_classifier

A simply news recommendation system. It parses International news headlines from New York Times and filter news that are classified as not likely to be read by user <br/>

A training data set is stored in a zip file in *data/data.zip* <br/>. To update training data set, please update these files here <br/>
#Required <br/>
Docker
<br/>

# Training data set <br/>

Should be stored in *data/data.zip*. There are 2 files <br/>

1.pos.jl : Each lines contains a json object of an article **likely to** be read <br/>
2.neg.jl : Each lines contains a json object of an article **unlikely** to be read <br/>

# How to set up <br/>

1. Pull docker image *dataquestio/python2-starter*, which contains a pre-configured environment for data processing (including scrapy, numpy, scipy, sklearn....etc) <br/>

2. Run Jupyter local server *docker run -d -p 8888:8888 -v /home/foo/dir:/home/ds/notebooks dataquestio/python2-starter* with /home/foo/dir is the project root dir. It will loads source code onto the jupyter local server at *localhost:8888* <br/>

3. Launch docker terminal of the container  *docker exec -it CONTAINER_ID /bin/bash*, with CONTAINER_ID is the container id of the image being run . You should get this return after the previous command.<br/>

4. In the new terminal, install *pip install -U pytest* to install pytest in your container <br/>

5. Launch localhost:8888 with your web browser, then a Jupyter interface is presented. Execute *launch_crawling.py* to get recommended New York Times article <br/>

# Pytest <br/>

In Jupyter terminal, launch *test.py -v py.test*
