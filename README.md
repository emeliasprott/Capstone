
# Decoding the California Legislative Process

This project collects, cleans, and analyzes data related to California’s legislative activities. The goal is to provide clear insights into the lawmaking process, including the development of bills, the final legislative outcomes, and the financial factors influencing decision-making.

## I. Introduction

## II. Data

### a. Legislative Information (*ca_leg*)

- Download ZIP files containing .dat and .lob files from official California state sources.
- Use a Docker container with a PostgreSQL database to extract data from the .dat and .lob files.
- Ensure .lob files are Unicode encoded and parse them as XML files into JSON format.
- Store the extracted data in CSV files

### b. Final Histories (*pdf_parsing*)

### c. Lobbying and Campaign Finance (*calaccess*)

The California Automated Lobbyist and Campaign Contribution and Expenditure Search System (CAL-ACCESS) is a database maintained by the California Secretary of State. It tracks campaign finance and lobbying activities, providing financial information supplied by state candidates, donors, lobbyists, and others.


## III. NLP

## IV. Graph Construction

## V. Graph-Based Temporal Neural Network
