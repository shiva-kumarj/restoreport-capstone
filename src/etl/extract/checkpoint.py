import psycopg2
import logging
import os

log_file_path = os.path.abspath("D:\My-Projects\stonecap\logs\lemmatization.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Checkpoint:
    def __init__(self, host, db, user, password):
        self.host = host
        self.db = db
        self.user = user
        self.password = password

    def create_checkpoint(self):
        # port = 5432
        table_name = "review_checkpoints"

        logging.info("checkpoint: Trying to establish connection with Db..")
        conn = psycopg2.connect(
            host=f"{self.host}",
            database=f"{self.db}",
            user=f"{self.user}",
            password=f"{self.password}",
        )
        if conn.closed == 0:
            logging.info("checkpoint: Connection Established")
        else:
            logging.info("checkpoint: Could not connect to DB")

        logging.info("checkpoint: Creating a 'checkpoint' table")
        cursor = conn.cursor()
        cursor.execute(
            f"""CREATE TABLE IF NOT EXISTS {table_name} (id SERIAL PRIMARY KEY, checkpoint INTEGER)"""
        )

        cursor.close()

        conn.commit()
        conn.close()

    def save_checkpoint(self, checkpoint):
        conn = psycopg2.connect(
            host=f"{self.host}",
            database=f"{self.db}",
            user=f"{self.user}",
            password=f"{self.password}",
        )

        cursor = conn.cursor()
        insert_query = (
            "INSERT INTO review_checkpoints (checkpoint) VALUES (%s) RETURNING id"
        )

        cursor.execute(insert_query, (checkpoint,))
        conn.commit()
        logging.info(f"checkpoint: Save checkpoint Successful")
        cursor.close()
        conn.close()

    def load_checkpoint(self):
        conn = psycopg2.connect(
            host=f"{self.host}",
            database=f"{self.db}",
            user=f"{self.user}",
            password=f"{self.password}",
        )
        cursor = conn.cursor()
        cursor.execute("select id, checkpoint from review_checkpoints ORDER BY id ASC")
        result = cursor.fetchall()
        if not result:
            result = [(0, 0)]
        conn.commit()
        logging.info(f"checkpoint: Loading checkpoint Successful")
        cursor.close()
        conn.close()

        return result
