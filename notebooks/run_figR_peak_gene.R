options(digits=5, max.print=100)  # Adjust numbers as needed
set.seed(123)

library(dplyr)
library(FNN)
library(chromVAR)
library(doParallel)
library(BuenColors)
library(FigR)
library(BSgenome.Hsapiens.UCSC.hg38)
atac_short = readRDS('atac_short.rds')
rna_short  = readRDS('rna_short.rds')
cisCorr <- FigR::runGenePeakcorr(ATAC.se = atac_short,
                           RNAmat = rna_short,
                           genome = "hg38", # One of hg19, mm10 or hg38 
                           nCores = 20,
                           p.cut = NULL, # Set this to NULL and we can filter later
                           n_bg = 100)
write.csv(cisCorr, "cisCorr.csv", row.names = TRUE)
