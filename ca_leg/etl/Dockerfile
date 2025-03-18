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
# COPY etl_data/ /etl_data/
COPY etl_data/batch2 /app/etl_data/
# COPY etl_data/pubinfo_2025.zip /etl_data/pubinfo_2025.zip
COPY etl_data/format_files /app/format_files

RUN chmod +x /app/run_etl.sh

CMD ["/bin/bash", "-c", "./run_etl.sh"]