#!/bin/bash
set -e # exit on error
# DIRECTORIES AND ENVIRONMENT
BAT_DIR="/etl_data/format_files"
TABLES_LC="$BAT_DIR/tables_lc.lst"
TABLES_UC="$BAT_DIR/tables_uc.lst"
NUM_PROCS=$(nproc || echo 4)
TEMP_DIR="/etl_data/temp"
OUTPUT_DIR="/etl_data/LOB"
mkdir -p "${TEMP_DIR}"
mkdir -p "${OUTPUT_DIR}"

# YEARS
YEARS="2021 2023 2025"
PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS lo;"
# loading .dat files
copy_file() {
    local file="$1"
    local table_lc="$2"
    local target_table="$3"
    PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" \
        -c "\COPY ${target_table} FROM '${file}' WITH (FORMAT csv, DELIMITER E'\t', HEADER false, NULL 'NULL', QUOTE '\`', ESCAPE '\\');"
}
export -f copy_file
export DB_PASSWORD DB_HOST DB_USER DB_NAME
# loading .lob files and converting to text
process_lob() {
    local data_dir="$1"
    local year="$2"
    local output_dir="$3"
    local lob_dir="${OUTPUT_DIR}/lob_files_${year}"
    local bill_analysis_dir="${lob_dir}/bill_analysis"
    local bill_version_dir="${lob_dir}/bill_version"
    mkdir -p "${bill_analysis_dir}"
    mkdir -p "${bill_version_dir}"
    find "${data_dir}" -type f -name "BILL_ANALYSIS_TBL_*.lob" -exec cp {} "${bill_analysis_dir}/" \;
    find "${data_dir}" -type f -name "BILL_VERSION_TBL_*.lob" -exec cp {} "${bill_version_dir}/" \;
    (cd "${lob_dir}" && zip -r "${output_dir}/${year}_lob_files.zip" .)
    rm -rf "${lob_dir}"
}

export -f process_lob
export DB_PASSWORD DB_HOST DB_USER DB_NAME
for YEAR in ${YEARS}; do
    ZIP_FILE="/etl_data/batch2/pubinfo_${YEAR}.zip"
    echo "Processing ${ZIP_FILE}..."
    DATA_DIR="${TEMP_DIR}/pubinfo_${YEAR}"
    unzip -q "${ZIP_FILE}" -d "${DATA_DIR}"

    paste "${TABLES_LC}" "${TABLES_UC}" | while read -r table_lc table_uc; do
        TARGET_TABLE="legislation_db.${table_lc}_${YEAR}"
        echo "Creating table ${TARGET_TABLE} and loading .dat file..."
        PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" \
            -c "DROP TABLE IF EXISTS ${TARGET_TABLE}; CREATE TABLE ${TARGET_TABLE} (LIKE legislation_db.${table_lc} INCLUDING ALL);"
        dat_file=$(find "${DATA_DIR}" -type f -name "${table_uc}.dat" 2>/dev/null | head -1)
        if [[ ! -s "${dat_file}" ]]; then
            echo "Warning: ${dat_file} is empty or does not exist. Skipping main file for ${table_lc}..."
        else
            echo "Splitting and loading data from ${dat_file}..."
            temp_split_dir=$(mktemp -d)
            split -n l/"${NUM_PROCS}" "${dat_file}" "${temp_split_dir}/part_"
            echo "Loading data in parallel..."
            find "${temp_split_dir}" -name "part_*" | parallel -j "${NUM_PROCS}" copy_file {} "${table_lc}" "${TARGET_TABLE}"
            echo "Cleaning up temporary files..."
            rm -rf "${temp_split_dir}"
        fi
    done

    process_lob "${DATA_DIR}" "${YEAR}" "${OUTPUT_DIR}"
    find "${TEMP_DIR}" -type f -delete
    echo "Finished processing ${ZIP_FILE}"
done
echo "ETL process complete."