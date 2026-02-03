# Data ETL Process — Step-by-Step Guide

This document walks through how to run the ETL process that ingests California Legislative Information ZIP archives into PostgreSQL using Docker. The process is batch-based by design to avoid memory and disk pressure on local machines.

## Overview

1. Install Docker Desktop so your machine can run PostgreSQL and the ETL inside containers.
2. Download the `leginfo` ZIP archives (do **not** unzip them).
3. Organize the ZIP files into batches (e.g. `batch1`, `batch2`, …) to avoid resource limits.
4. Update the ETL configuration (Dockerfile + shell script) to select the batch to load.
5. Rebuild the ETL container whenever the Dockerfile or `run_etl.sh` changes.
6. Run the ETL container to load data into PostgreSQL.
7. Keep Docker running and then execute the two notebooks:
   - `save-data.ipynb`
   - `xml_parse.ipynb`

---

## 2) Install Docker Desktop (First-Time Setup)

### A) macOS (Intel or Apple Silicon)

1. Go to <https://www.docker.com/products/docker-desktop/>.
2. Click **Download for Mac**.
3. Open the `.dmg` and drag **Docker** into Applications.
4. Launch Docker from Applications.
5. Follow the onboarding prompts and grant permissions if requested.
6. Wait until Docker finishes starting (the whale icon appears in the menu bar).

### B) Windows 10 / 11

1. Go to <https://www.docker.com/products/docker-desktop/>.
2. Click **Download for Windows** and run the installer.
3. Enable **WSL 2** if prompted.
4. Reboot if required.
5. Open Docker Desktop.
6. Wait until Docker shows **“Docker Desktop is running.”**

### C) Confirm Docker Works

Open a terminal (PowerShell on Windows, Terminal on macOS) and run:

```bash
docker --version
````

You should see a version string such as `Docker version 24.x.x`.

## 3) Prepare the ETL Data Folders

### A) Download the ZIP Files (Do **Not** Unzip)

1. Visit `leginfo.legislature.ca.gov`.
2. Download the `pubinfo_*.zip` archives you need.
3. Do **not** open or unzip the files—the ETL reads them directly.

### B) Create Batch Folders

Large batches can cause the ETL to fail, so keep each batch under ~5GB.

1. Navigate to `ca_leg/etl_data/`.
2. Create folders such as:

   * `batch1`
   * `batch2`
   * `batch3`

Example directory structure:

```text
ca_leg/
  etl_data/
    batch1/
    batch2/
    batch3/
```

### C) Split ZIPs Across Batches

Suggested approach:

- Group older years together (they are usually smaller).
- Put large, recent years in their own batches.
- Keep each batch below ~5GB.

**Note:** A full historical load will require running the ETL once per batch.

## 4) Update `run_etl.sh` (Daily Files Optional)

Open `ca_leg/etl/run_etl.sh` and locate this line:

```bash
# ZIP_FILES+=" $(ls "${BATCH_DIR}"/pubinfo_daily_*.zip 2>/dev/null || true)"
```

If you want to include daily ZIP files, remove the `#` comment and save the file.

## 5) Update the Dockerfile to Point to the Correct Batch

Open `ca_leg/etl/Dockerfile` and find the section similar to:

```dockerfile
COPY etl_data/batch3 /etl_data/batch3
ENV BATCH_DIR=/etl_data/batch3
```

To load a different batch, update **both lines**. For example, for `batch1`:

```dockerfile
COPY etl_data/batch1 /etl_data/batch1
ENV BATCH_DIR=/etl_data/batch1
```

**Important:** For a full load, you will repeat this step for each batch.

---

## 6) Rebuild Docker Containers After Any Script or Dockerfile Change

Any modification to `run_etl.sh` or the `Dockerfile` requires a full rebuild.

From the `ca_leg/` directory:

```bash
docker compose down --volumes --remove-orphans
docker compose build --no-cache
docker compose up --build
```

### What These Commands Do

- `down --volumes --remove-orphans`: removes containers, networks, and database volumes.
- `build --no-cache`: forces a clean rebuild of the ETL image.
- `up --build`: recreates and starts the containers.

The ETL container may exit once it finishes. The database container will continue running.


## 7) Run the ETL (For Each Batch)

Once the containers are built, the ETL runs automatically because `docker-compose.yml` calls `/app/run_etl.sh`.

To monitor progress:

```bash
docker compose logs -f etl
```

When complete, you should see:

```text
ETL process complete.
```

### Running the Next Batch

1. Stop containers (recommended):

   ```bash
   docker compose down
   ```

2. Update the Dockerfile to point to the next batch.
3. Rebuild and rerun using the commands in Section 6.

Repeat until all batches are processed.
> [!NOTE]
> Once the sh portion of ETL is finished, the container holds a complete SQL database. This can be launched independently at any time to access all of the legislative data simultaneously.

## 8) Keep Docker Running and Run the Notebooks

After the ETL finishes, keep Docker running and move on to the notebooks.

### A) Python Environment Setup (One-Time)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install jupyterlab pandas psycopg2-binary tqdm unidecode
```

### B) Launch Jupyter

```bash
jupyter lab
```

Open the URL shown in the terminal if it does not open automatically.

### C) Run the Notebooks (In Order)

1. `ca_leg/etl/save-data.ipynb`
2. `ca_leg/etl/xml_parse.ipynb`

**Important:** The PostgreSQL container must be running while these notebooks execute.

---

## 9) Stopping and Restarting Docker

To stop containers:

```bash
docker compose down
```

To resume later:

```bash
docker compose up
```

---

## 10) Troubleshooting Tips

### ETL Stops Early or Fails

- Verify batch size is under ~5GB.
- Split ZIPs into smaller batches and rerun.

### Changes Don’t Take Effect

You must rebuild without cache:

```bash
docker compose down --volumes --remove-orphans
docker compose build --no-cache
docker compose up --build
```

### Daily ZIP Files Are Missing

Confirm this line is uncommented in `run_etl.sh`:

```bash
ZIP_FILES+=" $(ls "${BATCH_DIR}"/pubinfo_daily_*.zip 2>/dev/null || true)"
```
