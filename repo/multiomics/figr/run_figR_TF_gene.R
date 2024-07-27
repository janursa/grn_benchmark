options(digits=5, max.print=100)  # Adjust numbers as needed
set.seed(123)

library(dplyr)
library(FNN)
library(chromVAR)
library(doParallel)
library(BuenColors)
library(FigR)
library(BSgenome.Hsapiens.UCSC.hg38)

out_dir <- '../../output'

cisCorr.filt = read.csv(paste0(out_dir, "/infer/figr/grn/cisCorr.filt.csv"))
RNAmat.s = readRDS(paste0(out_dir, "/infer/figr/grn/RNAmat.s.RDS"))
dorcMat.s = readRDS(paste0(out_dir, "/infer/figr/grn/dorcMat.s.RDS"))
atac = readRDS(paste0(out_dir, "/scATAC/atac.rds"))
figR.d <- runFigRGRN(ATAC.se = atac, # Must be the same input as used in runGenePeakcorr()
                     dorcTab = cisCorr.filt, # Filtered peak-gene associations
                     genome = "hg38",
                     dorcMat = dorcMat.s,
                     rnaMat = RNAmat.s, 
                     nCores = 10)
write.csv(figR.d, paste0(out_dir, "/infer/figr/grn/figR.d.csv"))
