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
  1. Run database in 'pg-yelp-network'
  > docker run -it `
 -e POSTGRES_USER="root" `
 -e POSTGRES_PASSWORD="root" `
 -e POSTGRES_DB="yelp_db" `
 -v D:\My-Projects\CAPSTONE\yelp_postgres_db:/var/lib/postgresql/data `
 -p 5432:5432 `
 --network=pg-yelp-network `
 --name yelp `
 postgres:13
  2. Run pgadmin in the same network
  > docker run -it `
-e PGADMIN_DEFAULT_EMAIL="admin@admin.com" `
-e PGADMIN_DEFAULT_PASSWORD="root" `
-p 8080:80 `
--network=pg-yelp-network `
 --name yelp-admin `
dpage/pgadmin4

5. data ingesion script

python cleaning_business_dataset.py --user=root --password=root --host=localhost --port=5432 --db=yelp_db --table_name=business

6. Update Dockerfile and build docker image
  > docker built -t yelp_ingest:v001 .
  > docker run yelp_ingest:v001 --user=root --password=root --host=localhost --port=5432 --db=yelp_db --table_name=business

