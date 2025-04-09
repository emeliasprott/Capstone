
# Decoding the California Legislative Process

This project collects, cleans, and analyzes data related to California’s legislative activities. The goal is to provide clear insights into the lawmaking process, including the development of bills, the final legislative outcomes, and the financial factors influencing decision-making. This repository contains tools and data pipelines for building a structured, multi-source database of California legislative activity, campaign finance, and lobbying records from the past 25+ years. The project is part of Emelia Sprott’s Computational Social Science capstone, and aims to support predictive modeling and policy analysis, including bill passage prediction and influence mapping.

## I. Introduction

### Repository Structure

Capstone/\
├── ca_leg/\
│   ├── etl/                    # Downloaded zip files, PDFs, unprocessed XML\
│       ├── Dockerfile          # Dockerfile for PostgreSQL container\
│       ├── run_etl.sh          # Bash script to automate SQL loading\
│       ├── save-data.ipynb     # Notebook to save SQL data as CSVs\
│       └── xml_parse.ipynb     # Notebook to parse XML into JSON\
├── pdf_parsing/\
│   ├── data-collection-2.ipynb # Notebook to collect and process PDFs\
│   └── text_cleaning.ipynb     # Notebook to clean and standardize text\
├── combining.ipynb             # Notebook to combine data from different sources\
├── eda1.html                   # Exploratory data analysis project\
├── text-modeling.ipynb         # Creating text embeddings\
├── README.md\

## II. Data

### a. Legislative Information (*ca_leg*)

1. Start with raw, downloaded zip file directories
    1. The file for 2011 was corrupted
    2. Does not include daily updates → maintenance docs
2. Using docker (for mac/windows compatibility), bash, and a PostgreSQL container, unzip directories and load .dat files into SQL; save .lob files in a separate zip file
    1. Provided data loading script included a SQL script for table creation, used this with a few modifications (to translate from MSSQL to Postgre)
3. Use python to parse .lob files (in XML format), and save as json file
4. use python to save SQL data as CSVs

### b. Final Histories (*pdf_parsing*)

1. Downloaded relevant pages from senate and assembly final histories from the past ~25 years and combined into PDFs
2. Processed documents with OCR-scanning and text recognition in Adobe
3. Used a series of functions to automate parsing the PDFs into tabular data
4. Using fuzzy-string matching and agglomerative clustering, resolve all typos (for string matching later)
5. Save complete, corrected data as CSVs

### c. Lobbying and Campaign Finance (*calaccess*)

The California Automated Lobbyist and Campaign Contribution and Expenditure Search System (CAL-ACCESS) is a database maintained by the California Secretary of State. It tracks campaign finance and lobbying activities, providing financial information supplied by state candidates, donors, lobbyists, and others.

1. Data is mostly clean and available in CSV format
2. Donor/recipient name repair
    1. Use a combination of brute-force regex, spacy similarity, fuzzy string matching, and Jaccardian distance to match names/committees

## Next Steps

### NLP and Embedding

- [ ] Summarize and embed bill versions using transformer models
- [ ] Create topic maps linking lobbying efforts to legislative content
- [ ] Generate embeddings for lobbying firm descriptions and names for similarity-based analysis

### Graph Construction and Modeling

- [ ] Build a heterogeneous, time-aware graph
    - **Nodes**: Bills, Legislators, Committees, Donors, Lobbyists
    - **Edges**: Authored, Voted, Donated, Lobbied, Sponsored
- [ ] Use Graph Neural Networks (GNNs) to predict bill passage likelihood
- [ ] Analyze influence patterns via attention weights and node embeddings

## Notes and Known Issues

- The 2011 legislative SQL archive was corrupted and could not be processed.
- The original MSSQL schema has been adapted for PostgreSQL compatibility.
- OCR-extracted text contains minor noise, but has been mostly resolved through post-processing.
- Matching and reconciliation scripts are designed for robustness, with multiple fallbacks and hybrid similarity measures.


## Setup (Coming Soon)

Detailed setup instructions for:

- Running PostgreSQL + Docker pipeline
- Reproducing data parsing and matching
- Setting up NLP models for embeddings
