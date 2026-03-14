


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
   ```

3. **Configure paths** by editing `env.sh` to point to your local directories, then regenerate `env.yaml` (used by all Python scripts):

   ```bash
   bash generate_env_yaml.sh
   ```

   Re-run this whenever you change `env.sh`. Do **not** run it from SLURM jobs.

## Repository Structure

- **`scripts/`** – Scripts processing results and explanatory analysis.  
- **`src/`** – Source files 
- **`draw/`** – Assembled figures.  

## Authors
Jalil Nourisa
Antoine Passemiers