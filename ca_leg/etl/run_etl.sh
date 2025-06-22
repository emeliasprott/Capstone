#!/bin/bash
set -e # exit on error
# DIRECTORIES AND ENVIRONMENT
BAT_DIR="/etl_data/format_files"
TABLES_LC="$BAT_DIR/tables_lc.lst"
TABLES_UC="$BAT_DIR/tables_uc.lst"
NUM_PROCS=$(nproc || echo 4)
TEMP_DIR="/etl_data/temp"
OUTPUT_DIR="/etl_data/LOB"
mkdir -p "${TEMP_DIR}" "${OUTPUT_DIR}"
ZIP_FILES=$(ls "${BATCH_DIR}"/pubinfo_*.zip)
# DATABASE CONNECTION
PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS lo;"
# loading .dat files
copy_file() {
    local file="$1"
    local target_table="$2"
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
        -c "\COPY $target_table FROM '$file' WITH (FORMAT csv, DELIMITER E'\t', HEADER false, NULL 'NULL', QUOTE '\`', ESCAPE '\\');"
}
# loading .lob files and converting to text
process_lob() {
    local data_dir="$1"
    local year="$2"
    local output_dir="$3"
    local lob_dir="${OUTPUT_DIR}/lob_files_${year}"
    local bill_analysis_dir="${lob_dir}/bill_analysis"
    local bill_version_dir="${lob_dir}/bill_version"
    mkdir -p "${bill_analysis_dir}" "${bill_version_dir}"
    find "${data_dir}" -type f -name "BILL_ANALYSIS_TBL_*.lob" -exec cp {} "${bill_analysis_dir}/" \;
    find "${data_dir}" -type f -name "BILL_VERSION_TBL_*.lob" -exec cp {} "${bill_version_dir}/" \;
    (cd "${lob_dir}" && zip -r "${output_dir}/${year}_lob_files.zip" .)
    rm -rf "${lob_dir}"
}
export -f copy_file
export DB_PASSWORD DB_HOST DB_USER DB_NAME
for ZIP_FILE in ${ZIP_FILES}; do
    YEAR=$(basename "${ZIP_FILE}" | grep -o '[0-9]\{4\}')
    echo "Processing ${ZIP_FILE}..."
    DATA_DIR="${TEMP_DIR}/pubinfo_${YEAR}"
    unzip -q "${ZIP_FILE}" -d "${DATA_DIR}"

    paste "${TABLES_LC}" "${TABLES_UC}" | while read -r table_lc table_uc; do
        TARGET_TABLE="legislation_db.${table_lc}_${YEAR}"

        echo "Creating table $TARGET_TABLE"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
            -c "DROP TABLE IF EXISTS $TARGET_TABLE;
                CREATE TABLE $TARGET_TABLE (LIKE legislation_db.$table_lc INCLUDING ALL);"

        dat_file=$(find "$DATA_DIR" -type f -name "${table_uc}.dat" | head -n 1)
        if [[ ! -s "$dat_file" ]]; then
            echo "$dat_file is empty or missing - skipping"
            continue
        fi

        echo "Splitting $dat_file and loading in parallel"
        split_dir=$(mktemp -d)
        split -n l/"$NUM_PROCS" "$dat_file" "$split_dir/part_"
        find "$split_dir" -name "part_*" | parallel -j "$NUM_PROCS" copy_file {} "$TARGET_TABLE"
        rm -rf "$split_dir"
    done

    process_lob "$DATA_DIR" "$YEAR" "$OUTPUT_DIR"
    rm -rf "$DATA_DIR"
    echo "Finished $ZIP_FILE"
done
echo "ETL process complete."