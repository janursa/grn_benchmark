library(Gviz)
library(GenomicRanges)
library(rtracklayer)
library(cicero)
set.seed(2017)
packageVersion("cicero")

# get the peaks
cicero_data <- read.table(paste0("../output/cicero/scATAC/peaks.txt"), sep = "\t", header = FALSE)
input_cds <- make_atac_cds(cicero_data, binarize = TRUE)
input_cds <- monocle3::detect_genes(input_cds)
input_cds <- estimate_size_factors(input_cds)
input_cds <- preprocess_cds(input_cds, method = "LSI")
input_cds <- reduce_dimension(input_cds, reduction_method = 'UMAP', preprocess_method = "LSI")
# reduced dimension and cicero object
umap_coords <- reducedDims(input_cds)$UMAP
cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = umap_coords)
# read chromsize
chromsizes <- read.csv('../output/cicero/chromsizes.csv')
# actual run
conns <- run_cicero(cicero_cds, chromsizes)
# save all peaks and connections
all_peaks <- row.names(exprs(input_cds))
write.csv(x = all_peaks, file = "../output/cicero/all_peaks.csv")
write.csv(x = conns, file = "../output/cicero/connections.csv")