FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    postgresql-client \
    coreutils \
    parallel \
    locales \
    bash \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN rm -rf /app/etl_data/*
COPY etl/run_etl.sh /app/run_etl.sh
# Optional: restrict data copied to container
# CHANGE DATASETS USED
# COPY etl_data/batch /etl_data/batch
# ENV BATCH_DIR=/etl_data/batch
# COPY etl_data/batch2 /etl_data/batch2
# ENV BATCH_DIR=/etl_data/batch2
COPY etl_data/batch3 /etl_data/batch3
ENV BATCH_DIR=/etl_data/batch3

RUN chmod +x /app/run_etl.sh

# RUN chmod +x /app/run_etl.sh

# CMD ["/bin/bash", "-c", "./run_etl.sh"]