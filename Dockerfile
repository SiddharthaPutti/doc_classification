FROM python:3.9
COPY classifier.pkl
COPY requirements.txt

RUN pip install -r requirements.txt
CMD python classifier.py 