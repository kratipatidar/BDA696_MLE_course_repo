FROM ubuntu

# Get necessary system packages
RUN apt-get update \
  && apt-get install --yes \
     build-essential \
     libmysqlclient-dev \
     mariadb-client \
     python3 \
     python3-pip \
     python3-dev \
     python3-pymysql \
  && rm -rf /var/lib/apt/lists/*

# Copy over requirements
COPY ./requirements.txt /scripts/requirements.txt

RUN pip3 install --compile --no-cache-dir -r/ scripts/requirements.txt

# Copy over code
COPY final_project_features_sql.sql /scripts/final_project_features_sql.sql
COPY final_1.py /scripts/final_1.py
COPY final_project_bash.sh /scripts/final_project_bash.sh


CMD ["/scripts/final_project_bash.sh"]
