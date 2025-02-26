


# Supplementary Code for geneRNIB

This repository provides supplementary code for the **geneRNIB** manuscript.  
For more details on geneRNIB, visit the main repository:  
[**github.com/openproblems-bio/task_grn_benchmark**](https://github.com/openproblems-bio/task_grn_benchmark).

## Setup Instructions

To run this repository, follow these steps:

1. **Download geneRNIB** and place it alongside this repository (`task_grn_inference`).
2. **Sync the results** within `task_grn_inference` using the following command:

   ```bash
   aws s3 sync s3://openproblems-data/resources/grn/results/ resources/results --no-sign-request

## Repository Structure

- **`notebooks/`** – Jupyter notebooks for processing results and explanatory analysis.  
- **`src/`** – Source files used in the notebooks.  
- **`draw/`** – Assembled figures.  

## Authors
Jalil Nourisa
