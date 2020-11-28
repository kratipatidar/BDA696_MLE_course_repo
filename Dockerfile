FROM ubuntu



# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy over code
COPY assignment_5_sql.sql assignment_5_sql.sql
COPY assignment_5_bash.sh assignment_5_bash.sh

# Create an unprivileged user
RUN useradd --system --user-group --shell /sbin/nologin services

# Switch to the unprivileged user
USER services

# Run app
CMD assignment_5_bash.sh  



