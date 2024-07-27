options(digits=5, max.print=100)  # Adjust numbers as needed
set.seed(123)

# library(Pando)
library(Seurat)
library(BSgenome.Hsapiens.UCSC.hg38)
library(Signac)
library(EnsDb.Hsapiens.v86)

library(Matrix)
#-------- import atac-seq count matrix and metadata and creat assay
X <- readMM("X_matrix.mtx")
X <- t(X)
annotation_peak <- read.csv("annotation_peak.csv", row.names = 1)
annotation_cells <- read.csv("annotation_cells.csv", row.names = 1)

# Filter out entries where seqname is not chr 
filter_indices <- grepl("^chr", annotation_peak$seqname)
annotation_peak_filtered <- annotation_peak[filter_indices, ]

# Filter the rows in X
X_filtered <- X[filter_indices, ]
  
peaks_matrix <- X_filtered
colnames(peaks_matrix) <- annotation_cells$obs_id 

rownames(peaks_matrix) = paste(annotation_peak_filtered$seqname, annotation_peak_filtered$ranges, sep = "_")
atac_assay <- CreateChromatinAssay(counts = peaks_matrix, ranges=GRanges(annotation_peak_filtered$seqname,
                              IRanges(annotation_peak_filtered$ranges)), 
                             colData = DataFrame(annotation_cells), genome = "hg38")
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- "UCSC"
genome(annotations) <- "hg38"
Annotation(atac_assay) <- annotations
#-------- import rna-seq count matrix (with row and col names already assigned ) and create seurat object
rna_short  = readRDS('rna_short.rds')
seurat_object <- CreateSeuratObject(count = rna_short, project = "pbmc", min.cells = 3, min.features = 200)
# add peaks
seurat_object[["peaks"]] = atac_assay

seurat_object