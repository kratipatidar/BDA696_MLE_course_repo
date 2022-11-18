FROM ubuntu

# Get necessary system packages
RUN apt-get update \
  && apt-get install --yes \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Copy over code
COPY assignment_5_sql.sql /scripts/assignment_5_sql.sql
COPY assignment_5_bash.sh /scripts/assignment_5_bash.sh

CMD ["/scripts/assignment_5_bash.sh"]
