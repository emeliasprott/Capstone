# Decoding the California Legislative Process

*A graph-based, self-supervised modeling system for understanding bills, stakeholders, and influence in California lawmaking.*

This repository contains the full pipeline for a computational social science capstone project that reconstructs the California legislative process as a heterogeneous graph and trains a graph neural network (GNN) to model procedural flow, policy similarity, and political influence.

The project integrates raw legislative data, historical PDFs, campaign finance and lobbying disclosures, and NLP embeddings to create a unified, machine-readable representation of how law is made in California.

Links to external presentations summarizing intermediate project stages:

* **Preliminary Data & EDA Overview:** [https://ca-leg-eda.my.canva.site/preliminary](https://ca-leg-eda.my.canva.site/preliminary)
* **Full EDA Presentation:** [https://ca-leg-eda.my.canva.site](https://ca-leg-eda.my.canva.site)
* **GNN Presentation:** [https://ca-leg-eda.my.canva.site/emelia-sprott](https://ca-leg-eda.my.canva.site/emelia-sprott)

---

## Overview of the Project Pipeline

The repository is structured around five core phases:

1. **Data Collection and Extraction**
2. **Data Cleaning, Standardization, and Integration**
3. **Graph Construction with Node and Edge Types**
4. **Self-Supervised GNN Development (PyTorch Geometric)**
5. **Analysis and Dashboard for Insights**

Each phase is implemented with reproducible notebooks, Python scripts, and R code.

---

## Repository Structure

### `ca_leg/etl/` – Legislative Data ETL Pipeline

These scripts process raw archives from the California Legislative Information system.

| File              | Description |
| ----------------- | --------------------------------------------------------------------- |
| `Dockerfile`      | Creates the PostgreSQL container for loading `.dat` legislative files |
| `run_etl.sh`      | Automated script to unzip raw archives and load them into SQL         |
| `save-data.ipynb` | Exports cleaned SQL tables to CSV for downstream analysis             |
| `xml_parse.ipynb` | Converts `.lob` XML files into JSON format                            |

### `pdf_parsing/` – Final Histories Extraction

Handles OCR and text extraction of Senate and Assembly Final Histories PDFs.

| File                    | Description |
| ----------------------- | ------------------------------------------------------- |
| `data-collection.ipynb` | Collects and parses PDFs from the past ~25 years        |
| `text_cleaning.ipynb`   | Cleans OCR text and resolves formatting inconsistencies |

### `dashboard/shiny/` – Legislative Dashboard

| File    |Description |
| ------- | ---------------------------------------------------------------------------------------------- |
| `app.R` | R Shiny dashboard for bill search, topic drilldown, funding analysis, and stakeholder insights |

### Root Files

| File                    | Description |
| ----------------------- | -------------------------------------------------------------------- |
| `LeGNN4-5.py`             | Full heterogeneous GNN architecture and training loop                |
| `preprocessing.ipynb`   | Pipeline to construct the heterogeneous graph                        |
| `topic-embedding-clustering.ipynb` | Generates sentence-transformer embeddings for topics and text fields |
| `combining.ipynb`       | Combines cleaned datasets from multiple sources                      |
| `comps.ipynb`           | Computes metrics used in the dashboard  |
| `eda1.html`             | Exploratory data analysis |

---

## Detailed Explanation of the Entire Pipeline

### I. Data Sources

All data sources are official, public, or historically archived. They include:

#### Legislative Information (1999–2025)

* Downloaded from **California Legislative Information** (leginfo.legislature.ca.gov)
* Archives include `.dat` files, `.lob` XML files, committee records, bill versions, roll calls, and histories
* Daily update feeds not included; the 2011 archive is corrupted and excluded
* Data is extracted and standardized into a consistent SQL schema using Docker + PostgreSQL

#### Senate and Assembly Final Histories

* Approximately 25 years of PDFs
* OCR processed using Adobe text recognition
* Parsed to extract:

  * committee rosters
  * session summaries
  * legislator names
  * bill actions and referral paths

#### CAL-ACCESS

Raw campaign finance and lobbying activity data from [California Secretary of State, CAL-ACCESS](https://www.sos.ca.gov/campaign-lobbying/helpful-resources/raw-data-campaign-finance-and-lobbying-activity)

* Campaign finance contributions
* Lobbying registrations and expenditures
* Donor and recipient entities
* Standardized using:

  * Regex cleaning
  * Fuzzy matching (Jaro, Levenshtein, token set ratios)
  * Jaccard similarity
  * spaCy sentence embeddings

---

### II. Data Cleaning and Integration

After collection, the data undergoes a multi-step integration process.

#### Name and Entity Standardization

Across legislative archives, PDFs, and CAL-ACCESS, names appear inconsistently (e.g., "Asm. Smith", "Smith, John", "Assemblymember John Smith", initials, abbreviations).

Standardization methods include:

* Regular expression normalization
* Removal of titles ("Sen.", "Asm.", "Rep.")
* Conversion of `last, first` and `first last` to canonical form
* NLP similarity scoring
* Multi-source contextual checks (e.g., committee membership + vote patterns)

#### Merging Across Sources

Merged tables include:

* bill histories
* bill versions and digests
* final histories
* legislator terms and district data
* lobbying contacts
* campaign finance donations
* temporal relationships

Key challenges addressed:

* Multiple bill versions per session
* Changing legislator names across terms
* Missing or inconsistent committee names
* OCR noise in older PDFs
* Lobbying disclosures naming staffers instead of legislators

---

### III. Graph Construction

The processed data is transformed into a heterogeneous graph representing California policymaking.

#### Nodes

Nodes represent entities and actors:

* `bill`
* `bill_version`
* `legislator`
* `legislator_term`
* `committee`
* `donor`
* `lobby_firm`

#### Edges

Edges encode relationships and legislative actions:

* Authorship (`legislator_term → bill_version`)
* Committee referrals (`committee → bill_version`)
* Bill version ordering (`bill_version → bill_version`)
* Version–bill mapping (`bill_version → bill`)
* Roll-call votes
* Campaign contributions (`donor → legislator_term`)
* Lobbying contacts (`lobby_firm → legislator_term`)

#### Embeddings

All textual components are embedded using sentence-transformers:

* bill digests
* committee names
* legislator names
* donor and lobby firm names (where applicable)

These embeddings support entity matching, semantic similarity, and GNN initialization.

---

### IV. GNN Modeling (PyTorch Geometric)

The graph is converted into a `HeteroData` object and passed to a custom heterogeneous GNN.

#### Architecture

* Relation-aware message passing layers
* Type-specific linear transformations
* Multi-layer aggregation across the graph
* Encodes both structure and text features

#### Training Objectives (Self-Supervised)

No labels are required.
The model learns through:

* Link reconstruction (predicting masked edges)
* Feature denoising (reconstructing masked node features)
* Contrastive learning (encouraging nodes in similar contexts to cluster together)
* Topic alignment (integrating topic embeddings for semantic structure)

#### Learned Representations

The trained model embeds:

* Bills by policy area and procedural similarity
* Committees by issue domain and gatekeeping behavior
* Legislators by voting patterns, authorship, and stakeholder networks
* Donors and lobby firms by policy focus and alignment
* Bill versions by semantic and procedural identity

#### Downstream Uses

Embeddings support:

* Bill similarity search
* Stakeholder alignment maps
* Topic clusters
* Influence modeling
* Committee gatekeeping analysis
* Legislative trajectory prediction
* Dashboard-ready summaries

---

### V. Dashboard and Analysis

The R Shiny dashboard (`dashboard/shiny/app.R`) provides an interpretive layer over the trained embeddings.

Features include:

* Bill search via embedding similarity
* Topic drilldowns with top bills, legislators, committees, donors, and lobby firms
* County-level funding maps via shapefiles
* Donor and lobby firm clustering
* Legislator influence and topic alignment
* Funding flows and partisan trends
* Session-over-time metrics

The dashboard integrates all cleaned data, model outputs, and derived metrics.

---

## Notes and Known Issues

* OCR introduces minor text errors in older PDFs; these are largely mitigated through cleaning.
* The original Microsoft SQL schema was adapted for **PostgreSQL** compatibility.
* Some lobbying disclosures list staffers instead of legislators; these were resolved via contextual matching where possible.
