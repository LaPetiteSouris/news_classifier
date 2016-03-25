FROM python:2.7

COPY . /src
WORKDIR /src

MAINTAINER Tung Hoang <sontunghoang@gmail.com>
# Prerequisites

RUN apt-get update -y
RUN apt-get install -y python-setuptools
RUN pip install --upgrade pip



#numpy 
RUN apt-get install -y --force-yes python-dev
RUN pip install numpy

#scipy 

RUN apt-get update
RUN apt-get install -y build-essential gfortran libatlas-base-dev 
RUN pip install scipy

RUN pip install -U scikit-learn



#security updates
RUN apt-get upgrade -y
