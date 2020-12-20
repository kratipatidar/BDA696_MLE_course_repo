#!/bin/bash
echo "Sleeping for 10 seconds"
sleep 10

echo "creating db"
mysql -h mariadb-service -u root -psecret -e "CREATE DATABASE IF NOT EXISTS baseball;"
echo "reading in file"
mysql -h mariadb-service -u root -psecret baseball < /data/baseball.sql
echo "running file"
mysql -h mariadb-service -u root -psecret baseball < /scripts/final_project_features_sql.sql

# saving results
echo "saving results"
mysql -h mariadb-service -u root -psecret baseball -e '
  SELECT * FROM final_baseball_features;' > /results/results.txt

python ./scripts/final_1.py
