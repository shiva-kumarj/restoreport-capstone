# Questions

## 1. Which is a better approach to build data pipeline?
1. load data into memory (python script), perform cleaning (fixing data types, anamolous data, etc) then ingest into a DB (postgres in this case)
2. Load the data into DB as it and perform cleaning using SQL and update the db (Schema) as the data evolves with more and more cleaning.
 
## Answer
The "better" approach can vary from one situation to another.

#### 1. Load data into memory, perform cleaning, then ingest into a DB:

**Pros:**

- More flexibility in data manipulation and cleaning as you can leverage Python libraries and tools.
- Easier to handle complex transformations that might be challenging with SQL alone.
- You can store intermediary or cleaned data in a format other than the database, which may be useful for various purposes.

**Cons:**

- Can be slower for very large datasets as all data needs to be loaded into memory.
- Requires additional code for data transformation and may lead to more complex scripting.


#### 2. Load data into DB as-is and perform cleaning using SQL:

**Pros:**

- Utilizes the database's processing power and indexing capabilities, which can be efficient for large datasets.
- Keeps the data transformation logic within the database, making it easier to maintain and manage as the data evolves.
- Often more suitable for real-time or near-real-time data ingestion.

**Cons:**

- Limited flexibility for complex data transformations compared to Python or other programming languages.
- May require altering the database schema frequently as the data evolves, which can be a maintenance challenge.

My situation:
1. The data is big, it takes long time to process.
2. Has to be batched to make it resistant against interruptions.
3. Filtering data takes a long time. 
4. Inner join of 2 tables takes even longer on memory. do able in a database like MongoDB.

A combination of both approaches may work. 
1. Do complex processing using Python.
2. Do memory intensive processing in database.



## 2. Snowflake vs (Airflow or Prefect)
Snowflake is a cloud based data warehousing solution. It provides SQL based quering. On the other hand Prefect and Airflow are workflow orchestration tools. So these two categories of tools cannot be compared. 

Workflows in Airflow are represented using DAGs (Directed Acyclic Graph). Where each node is a single unit of a task and the edges are the sequence of operations. 

Tasks are individual units of work. A task can have dependency of other tasks thus forming a DAG. Tasks in Airflow are represented by operators. Operators define the logic to execute a specific task. 



## 3. When ingesting multiple data files into the database(say fictional_db), my approach is to spin up multiple containers, where each performs some preprocessing of the data and ingests it into a table in fictional_db. Does this require some sort of a tool that orchestrates container creation and lifecycle or is Prefect enough to create a workflow that incorporates docker? or should my approach be to use one container and do all the ingestion?

The single container approach certainly simpler, more resource efficient, and easier to implement. 

#### Bird's eye view of the project

1. Prefect for workflow orchestration.
2. GCP(and it various services) for data lake(staging), data warehousing, data analysis.

#### Project Approach
1. Ingest data into Postgres db.
   1. `Container1` hosts PostgresDB that will hold the cleaned **business data** and **Review data**.
   2. The **Raw Review data** will be processed by **data ingestion** container and ingest **Review** table in batches.
   3. **Data Ingestion Container** must first ingest **Raw Business**, then process the **raw review** in batches (resume from checkpoints after stoppage.)
   4. The raw review data is too large to even load in memory at once. Loading and processing it in batches while maintaining checkpoints for safe recovery from failure.
   5. Performing data validity of each batch.
   6. `Container2` hosts the PostgresAdmin that provides UI and admin control to access the db in `Container1`.
   7. Both `Container1` and `Container2` are put into a shared Docker network (`docker network create`).
   8. A python script (clean_and_ingest_business.py, more datasets yet to come) reads the data from "staging area"(host machine for now), cleans it and ingests it into `Container1`.
   9. Dockerize the ingesion script (container must have access to the raw data)**(problem 1)**
   **Solution Problem 1**: As of now the raw data is stages in the host machine, so the container with the ingesion script can access the raw data from host machine by volume mounting a directory between the host and container.
   > docker run -v /path/on/host:/path/in/container your_image_name

   
   10. **When the container goes on cloud** I can leverage bind mounts to mount a directory from a cloud storage directly into the docker container.

   11. **If my usecase contains multiple containers** use docker compose to define a multi-container environment. specify volume mounts and services in the docker-compose.yml file. 

#### Question


#### Updates:
- [3/16]: Extract sentiment of full review text, not per sentence.
- [3/16]: Ditched Stanza for sentiment analysis, and adopted TextBlob instead. Highest Priority is speed of processing.
- [3/16]: Using pickle file to save checkpoints during review processing.
- [3/16]: Adopted Huggingface transformers for review labelling instead of Llamafile or openAI LLM.
- [3/16]: Separating data validation from data processing script. Better to first perform data cleaning and then validation after that.
- [3/21]: Huggingface is also very slow for text classification. Building my own model instead.
- [3/21]: Using GloVe's pretrained word embeddings for feature engineering input text.
- [3/21]: Built simple ML model using BalancedRandomForestClassifier (vanilla). Not good training or testing accuracy but able to predict very low represented classes. Better than other traditional models.
- [3/21]: Very fast prediction.
- [3/25]: Sequence of code execution. clean_and_ingest_business.py -> full_review_processing_textblob.py -> validate_data ->   
Resources:
1. GloVe: Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf] [bib]

