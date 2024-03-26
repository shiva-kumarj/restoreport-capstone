# Commands

1. Create container with Postgres
> docker run -it `
 -e POSTGRES_USER="root" `
 -e POSTGRES_PASSWORD="root" `
 -e POSTGRES_DB="yelp_db" `
 -v D:\My-Projects\CAPSTONE\yelp_postgres_db:/var/lib/postgresql/data `
 -p 5432:5432 `
 postgres:13

2. Connect pgcli to database (creates new container for pg database)
> pgcli -h localhost -p 5432 -u root -d yelp_db 

3. Connect pgAdmin to database (creates new container for pgadmin)
> docker run -it `
-e PGADMIN_DEFAULT_EMAIL="admin@admin.com" `
-e PGADMIN_DEFAULT_PASSWORD="root" `
-p 8080:80 `
dpage/pgadmin4

4. Docker network create
> docker network create pg-yelp-network
  
5. Run database in 'pg-yelp-network'
  > docker run -it `
 -e POSTGRES_USER="root" `
 -e POSTGRES_PASSWORD="root" `
 -e POSTGRES_DB="yelp_db" `
 -v D:\My-Projects\CAPSTONE\yelp_postgres_db:/var/lib/postgresql/data `
 -p 5432:5432 `
 --network=pg-yelp-network `
 --name pg-yelp-db `
 postgres:13

6. Run pgadmin in the same network
  > docker run -it `
-e PGADMIN_DEFAULT_EMAIL="admin@admin.com" `
-e PGADMIN_DEFAULT_PASSWORD="root" `
-p 8080:80 `
--network=pg-yelp-network `
--name yelp-pgadmin `
dpage/pgadmin4

7. data ingesion script

> python clean_and_ingest_business.py --user=root --password=root --host=localhost --port=5432 --db=yelp_db --table_name=business

8. Update Dockerfile and build docker image
  > docker build -t yelp_ingest:v001 .

9. Mount the data directory into the container to access the data.
  > docker run -v D:\My-Projects\CAPSTONE\data\raw:/data yelp_ingest:v001

10. Docker run command
  > docker run -v D:\My-Projects\CAPSTONE\data\raw:/raw_data --network=stonecap_default yelp_ingest:v001 --user=root --password=root --host=pgdatabase --port=5432 --db=yelp_db --table_name=business



