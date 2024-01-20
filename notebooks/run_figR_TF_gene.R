options(digits=5, max.print=100)  # Adjust numbers as needed
set.seed(123)

library(dplyr)
library(FNN)
library(chromVAR)
library(doParallel)
library(BuenColors)
library(FigR)
library(BSgenome.Hsapiens.UCSC.hg38)
cisCorr.filt = read.csv('cisCorr.filt.csv')
RNAmat.s = readRDS('RNAmat.s.RDS')
dorcMat.s = readRDS('dorcMat.s.RDS')
atac_short = readRDS('atac_short.rds')
figR.d <- runFigRGRN(ATAC.se = atac_short, # Must be the same input as used in runGenePeakcorr()
                     dorcTab = cisCorr.filt, # Filtered peak-gene associations
                     genome = "hg38",
                     dorcMat = dorcMat.s,
                     rnaMat = RNAmat.s, 
                     nCores = 30)
write.csv(figR.d, 'figR.d.csv')