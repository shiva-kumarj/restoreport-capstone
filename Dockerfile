FROM python:3.9

RUN 
RUN pip install pandas sqlalchemy psycopg2

WORKDIR /app
COPY clean_and_ingest_business.py clean_and_ingest_business.py  

ENTRYPOINT ["python", "clean_and_ingest_business.py"]