
# Decoding the California Legislative Process

## I. Introduction

This project builds a comprehensive graph-based representation of the California legislative process, combining diverse datasets to model relationships among bills, legislators, committees, donors, lobbyists, and more. The goal is to provide clear insights into the lawmaking process, including the development of bills, final legislative outcomes, and the financial factors influencing legislative decision-making.

The repository is designed to enable future users (researchers, policy analysts, developers) to extend, re-train, or use the model for new applications such as bill search, stakeholder analysis, and legislative forecasting.

The project is structured into multiple phases:

1. Data collection and extraction
2. Data cleaning and integration
3. Graph construction
4. GNN modeling
5. Analysis and Insights

### Repository Structure

#### Capstone/

- ca_leg/
  - etl/                    # Downloaded zip files, PDFs, unprocessed XML
    - Dockerfile            # Dockerfile for PostgreSQL container
    - run_etl.sh            # Bash script to automate SQL loading
    - save-data.ipynb       # Notebook to save SQL data as CSVs
    - xml_parse.ipynb       # Notebook to parse XML into JSON
- pdf_parsing/
  - data-collection.ipynb   # Notebook to collect and process PDFs
  - text_cleaning.ipynb     # Notebook to clean and standardize text
- combining.ipynb           # Notebook to combine data from different sources
- eda1.html                 # Exploratory data analysis project
- graph-construction.ipynb  # Notebook to construct the graph
- LeGNN.py                  # Script to create and train the GNN
- README.md                 # Current file

## II. [Data](https://ca-leg-eda.my.canva.site/preliminary) (*links to presentation*)

### a. Data Sources

The project combines multiple official and public data sources:

- [California Legislative Information](https://leginfo.legislature.ca.gov) raw zip files
- [Senate and Assembly Final Histories](https://clerk.assembly.ca.gov/archive-list) PDFs
- [CAL-ACCESS](https://www.cal-access.org)
  - [Campaign Finance](https://powersearch.sos.ca.gov/index.php)
  - [Lobbying](https://www.sos.ca.gov/campaign-lobbying/helpful-resources/raw-data-campaign-finance-and-lobbying-activity)

#### Legislative Information (*ca_leg*)

1. Start with raw, downloaded zip file directories from leginfo.gov
   - Note: The file for 2011 was corrupted and daily updates are not included.
2. Use Docker (for Mac/Windows compatibility), bash, and a PostgreSQL container to unzip directories and load .dat files into SQL; .lob files are saved in a separate zip archive.
3. Apply SQL schema scripts (modified for PostgreSQL compatibility) to load data.
4. Use Python to parse .lob XML files into JSON.
5. Export SQL tables into CSV format for downstream processing.

#### Final Histories (*pdf_parsing*)

1. Download Senate and Assembly Final Histories PDFs from the past ~25 years.
2. Process documents using OCR and Adobe text recognition.
3. Automate parsing and extraction of committee rosters and legislator names into tabular data.
4. Clean and resolve typos using fuzzy matching and agglomerative clustering to prepare for entity linking.
5. Save clean, corrected data as CSVs.

#### Lobbying and Campaign Finance (*calaccess*)

The California Automated Lobbyist and Campaign Contribution and Expenditure Search System (CAL-ACCESS) provides donor, lobbyist, and contribution records.

1. Data is clean and readily available in CSV format.
2. Standardize donor and recipient names:
    - Use regex, SpaCy similarity scoring, fuzzy string matching, and Jaccard distance methods to resolve duplicates and inconsistencies.

### b. Data Processing

1. Merge all datasets (bills, legislators, committees, donors, lobbyists).
2. Perform NLP-based and fuzzy matching to identify relationships and unify entity names.
3. Resolve inconsistencies using multi-dimensional relationship matching (e.g. matching based on correlated features across different datasets).

#### [EDA](https://ca-leg-eda.my.canva.site) (*links to presentation*)

### III. Graph Construction

The graph construction phase transforms cleaned legislative data into a rich, structured network. This graph is the core representation of legislative relationships and is built to capture both procedural flow and political influence.

#### Generating Embeddings

- All key text fields — bill text, committee names, and legislator names — are encoded using sentence embeddings.
- These embeddings help resolve inconsistencies across datasets, linking entities that may be formatted differently but refer to the same item.

#### Entity Matching and Resolution

- Entity matches are refined using multi-dimensional similarity checks, combining features such as text similarity, relational context, and metadata.
- This ensures robust connections, especially where data sources disagree or have missing values.

#### Building the Graph

- Uses an object-oriented graph builder to iteratively add nodes and edges.
- Node types include bills, bill versions, legislators, committees, lobby firms, and donors.
- Edge types represent important legislative actions and relationships, such as:
  - Authorship (who authored a bill)
  - Voting (who voted on what)
  - Committee referrals (committee assignments)
  - Lobbying (lobby firm efforts)
  - Donations (campaign contributions)

### [GNN](https://ca-leg-eda.my.canva.site/emelia-sprott) (*links to presentation*)

The GNN transforms the graph into a format suitable for machine learning and trains a heterogeneous GNN to produce meaningful embeddings.

#### Preparing the Graph

- Converts the assembled graph into PyTorch Geometric's HeteroData format.
- Supports heterogeneous node and edge types for fine-grained modeling.

#### Model Design

- Relation-aware message passing layers handle the complexity of legislative relationships.
- Uses modular architecture for flexibility and scalability.

#### Self-Supervised Training

- The model does not rely on labels; instead, it learns through self-supervised objectives:
- Link reconstruction: Predict missing or masked connections.
- Feature denoising and reconstruction: Encourage robustness and representation quality.
- Contrastive clustering: Group bills and stakeholders by latent policy topics and alignments.

#### What the Model Learns

- Captures semantic similarity (e.g., bills on related topics).
- Learns procedural and political relationships (e.g., donor influence or legislator alliances).
- Embeds stakeholder dynamics (e.g., which actors cluster around policy areas).

#### Outputs

The trained GNN produces embeddings that enable downstream legislative intelligence tasks:

- Bill similarity search based on content and context.
- Stakeholder alignment analysis, revealing connections between donors, lobbyists, committees, and legislators.
- Policy clustering and topic exploration, allowing unsupervised discovery of major themes.

## Analysis

## Notes and Known Issues

- The 2011 legislative SQL archive was corrupted and could not be processed.
- The original MSSQL schema has been adapted for PostgreSQL compatibility.
- OCR-extracted text contains minor noise, but has been mostly resolved through post-processing.
