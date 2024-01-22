FROM python:3.9

RUN 
RUN pip install pandas sqlalchemy psycopg2

WORKDIR /app
COPY clean_and_ingest_business.py clean_and_ingest_business.py  

RUN mkdir -p /cleaned_data

ENTRYPOINT ["python", "clean_and_ingest_business.py"]