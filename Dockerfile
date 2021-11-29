FROM ubuntu:18.04
COPY flaskapp.py /exp/assg_10.py
COPY requirements.txt /exp/requirements.txt
COPY model1svm.joblib /exp/model1svm.joblib
COPY modeldecisiontree.joblib /exp/modeldecisiontree.joblib
RUN apt-get update && apt-get install -y python3.7 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
RUN apt-get install -y curl
WORKDIR /exp
ENV LANG='C.UTF-8' LC_ALL='C.UTF-8'
ENV FLASK_APP flaskapp
CMD ["flask", "run"]
