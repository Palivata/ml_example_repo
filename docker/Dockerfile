FROM ubuntu:22.04

RUN apt-get update && apt-get install -y software-properties-common gcc
RUN apt-get update && apt-get install -y python3.10 python3-distutils python3-pip python3-apt ffmpeg libsm6 libxext6
RUN pip install -U pip --no-cache-dir
RUN pip install --no-cache-dir ipywidgets
COPY ../requirements.txt requirements.txt
COPY clearml.conf /root/clearml.conf
RUN pip install -r requirements.txt --no-cache-dir
WORKDIR /src
ENV PYTHONPATH=/src
ENTRYPOINT ["/bin/bash"]
