FROM ubuntu:18.04
COPY assignment2 /exp/mlops
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3.py
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
