## Which is a better approach to build data pipeline?
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


