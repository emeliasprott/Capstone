# Decoding the California Legislative Process

## Table of Contents

- [Project Overview](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Data Coverage and Sources](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
    - [Legislative Records (1980–2025)](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
    - [Senate and Assembly Final Histories (PDFs)](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
    - [Campaign Finance and Lobbying Data (CAL-ACCESS)](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Pipeline Overview](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Data Integration and Entity Resolution](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Graph Representation](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
    - [Node Types](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
    - [Edge Types and Semantics](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Features and Representations](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Graph Neural Network Model](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
    - [Model Characteristics](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
    - [Training Objective](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Outputs and Intended Use](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Limitations and Known Issues](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Notes on Reproducibility](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)
- [Acknowledgments](https://www.notion.so/Decoding-the-California-Legislative-Process-2f7e716cc2478041907cfc3fd9de7013?pvs=21)



## Project Overview

This repository contains the core modeling and data-integration work for my computational social science capstone project that reconstructs the California legislative process as a temporal, heterogeneous graph and learns context-aware representations of bills, legislators, committees, donors, and lobbying firms using a self-supervised graph neural network (GNN).

The project integrates multiple public data sources, including legislative archives, committee records extracted from historical PDFs, and campaign finance and lobbying disclosures, into a unified relational representation of how policy is written, amended, voted on, and influenced in California. Rather than treating legislation as a flat text corpus or a sequence of roll calls, the system explicitly models procedure, institutional structure, and stakeholder relationships.

Supporting materials and presentations from intermediate project stages:

- **Preliminary Data & EDA Overview:** [https://ca-leg-eda.my.canva.site/preliminary](https://ca-leg-eda.my.canva.site/preliminary)
- **Full EDA Presentation:** [https://ca-leg-eda.my.canva.site](https://ca-leg-eda.my.canva.site/)
- **GNN Presentation:** [https://ca-leg-eda.my.canva.site/emelia-sprott](https://ca-leg-eda.my.canva.site/emelia-sprott)



## Data Coverage and Sources

### Legislative Records (1980–2025)

- Sourced from [**California Legislative Information**](http://leginfo.legislature.ca.gov/) archives and daily update files
- Includes bills, bill versions, actions, votes, committee referrals, and outcomes
- The pipeline supports both historical archive files and daily legislative feeds
- Earlier records (approximately 1980–2001) exhibit higher formatting inconsistency and missingness than later years

### Senate and Assembly Final Histories (PDFs)

- Approximately 25 years of Senate and Assembly Final Histories
- Used to recover committee structures, rosters, and procedural information not reliably encoded in raw legislative files
- PDFs are OCR-processed and cleaned prior to integration
- Residual OCR noise is mitigated but not entirely eliminated

### Campaign Finance and Lobbying Data (CAL-ACCESS)

- Raw campaign contribution and lobbying activity data from the [California Secretary of State & CAL-ACCESS](https://www.sos.ca.gov/campaign-lobbying/helpful-resources/raw-data-campaign-finance-and-lobbying-activity)
- Includes donors, lobbying firms, expenditures, beneficiaries, and dates
- Extensive entity standardization is applied to:
    - normalize donor and firm names
    - resolve legislators across terms
    - match financial activity to the correct legislative recipients



## Pipeline Overview

The repository implements a multi-stage pipeline:

1. **SQL Ingestion and Normalization**
Legislative, financial, and lobbying records are ingested into a containerized PostgreSQL database that unifies records across all years.
2. **PDF Parsing and Committee Reconstruction**
Historical PDFs are parsed to reconstruct committee rosters, memberships, and institutional structure.
3. **Entity Resolution and Standardization**
Legislators, committees, donors, and firms are standardized across sources using rule-based cleaning, contextual matching, and similarity checks.
4. **Feature and Embedding Preparation**
Textual fields (e.g., bill digests, committee names, motion text) are converted into vector representations for downstream modeling.
5. **Graph Construction**
All entities and relationships are assembled into a heterogeneous, temporally ordered graph.
6. **Self-Supervised GNN Training**
A custom heterogeneous GNN is trained to learn relational embeddings without relying on labeled outcomes.



## Data Integration and Entity Resolution

A central challenge in this project is reconciling inconsistent, ambiguous, and typo-prone text fields across legislative records, PDFs, and financial disclosures.

Key aspects of the integration methodology include:

- Canonicalization of legislator and committee names across chambers and terms
- Construction of unified legislator–term records to prevent conflating identities across time
- Multi-stage matching of lobbying beneficiaries and campaign expenditure targets using:
    - exact and token-based matches
    - chamber- and term-aware heuristics
    - ranked fuzzy matching with confidence thresholds
    - explicit tagging of match methods to track reliability

This process produces cleaned, standardized linkages between financial activity and legislative actors that can be safely used in downstream modeling.


## Graph Representation

The legislative process is modeled as a heterogeneous graph in which nodes represent entities and edges represent institutional, procedural, and financial relationships. Temporal structure is preserved explicitly through bill versions, legislator terms, and dated interactions.

### Node Types

- **bill**
A legislative measure as an abstract entity across its lifecycle.
- **bill_version**
A specific version of a bill, capturing amendments, procedural requirements, and text at a given point in time.
- **legislator**
A person-level identity node representing an individual legislator across their career.
- **legislator_term**
A legislator within a specific chamber and legislative term. This allows voting behavior, committee assignments, and financial relationships to vary over time without collapsing a legislator’s entire career into a single node.
- **committee**
A legislative committee in a given term and chamber.
- **donor**
An entity making campaign contributions.
- **lobby_firm**
A registered lobbying organization.


### Edge Types and Semantics

Edges encode both structure and directionality. For most relationships, both forward and reverse edges are included to support symmetric message passing.

Key relationships include:

- **Versioning and procedure**
    - `bill_version → bill` (version membership)
    - `bill_version → bill_version` (temporal ordering of versions)
- **Authorship and sponsorship**
    - `legislator_term → bill_version` (authorship with role strength)
    - Committee sponsorship where applicable
- **Committee structure**
    - `legislator_term → committee` (membership with positional attributes)
    - `committee → bill / bill_version` (readings and referrals)
- **Voting behavior**
    - `legislator_term → bill_version` (votes with motion context and timestamps)
- **Financial influence**
    - `donor → legislator_term` (campaign contributions)
    - `lobby_firm → legislator_term / committee` (lobbying activity)
- **Identity linkage**
    - `legislator_term → legislator` (same-person relationships across terms)

This design allows the model to jointly reason about procedure, institutional roles, voting behavior, and financial relationships within a single relational structure.


## Features and Representations

Each node carries a mix of:

- **Text-derived features**
Vector representations of bill digests, subjects, committee names, and motion text.
- **Categorical encodings**
Party affiliation, chamber, committee position, vote thresholds, sponsorship roles.
- **Temporal attributes**
Dates of actions, votes, versions, and financial transactions.

Text representations are used as input features, not as standalone outputs. The GNN learns higher-level embeddings that integrate text with relational and procedural context.

## Graph Neural Network Model

The heterogeneous graph is converted into a PyTorch Geometric `HeteroData` object and passed to a custom GNN architecture (`LeGNN4-5.py`).

### Model Characteristics

- Relation-aware message passing across node and edge types
- Type-specific transformations for heterogeneous features
- Explicit handling of procedural order via version and action edges
- Designed to scale to large, sparse graphs

### Training Objective

The model is trained in a self-supervised manner. Rather than relying on labeled ideological scores or outcomes, it learns representations by:

- reconstructing voting relationships,
- aggregating procedural and financial signals across bill lifecycles,
- encouraging consistency and smoothness across bill versions and actor roles.

This approach allows the model to learn meaningful structure directly from observed legislative behavior and institutional context.

## Outputs and Intended Use

Because of data size constraints, outputs are not distributed via GitHub. When run locally, the pipeline produces:

- A unified PostgreSQL database spanning all years and data sources
- A relational graph representation of the legislative process
- Learned embeddings for bills, legislators, committees, donors, and firms

These embeddings support downstream analysis such as:

- policy similarity and clustering
- procedural trajectory analysis
- stakeholder alignment and influence mapping
- institutional role analysis (e.g., committee gatekeeping)

The system is designed for analysis and exploration, not for automated decision-making or policy prediction.

## Limitations and Known Issues

- Legislative data prior to approximately 2001 contains higher rates of missing or inconsistent records
- OCR-derived text from older PDFs may contain residual errors
- Lobbying disclosures sometimes reference staff or committees rather than individual legislators
- Financial relationships are observational and should not be interpreted causally

## Notes on Reproducibility

This repository documents methodology and modeling logic rather than serving as a lightweight, plug-and-play package. Full reproduction requires:

- access to large public data downloads
- local database setup
- significant compute and storage resources

Where possible, scripts are written to run end-to-end, but some steps (e.g., embedding generation) are intentionally modular.

## Acknowledgments

This project relies entirely on public data sources provided by the California Legislature and the California Secretary of State. Any errors or interpretations are the responsibility of the author.
