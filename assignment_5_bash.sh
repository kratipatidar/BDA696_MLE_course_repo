#!/bin/bash

mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS baseball;"
mysql -u root -p baseball < baseball.sql
mysql -u root -p baseball < assignment_5_sql.sql
