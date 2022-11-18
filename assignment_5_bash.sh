#!/bin/bash
echo "Sleeping for 10 seconds"
sleep 10


mysql -h mariadb-service -u root -psecret -e "CREATE DATABASE IF NOT EXISTS baseball;"
mysql -h mariadb-service -u root -psecret baseball < /data/baseball.sql
mysql -h mariadb-service -u root -psecret baseball < /scripts/assignment_5_sql.sql

# saving results
mysql -h mariadb-service -u root -psecret baseball -e '
  SELECT * FROM rolling_ba_100_days;' > /results/results.txt
