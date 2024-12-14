library(tidyverse)
library(funkyheatmap)

library(rprojroot)

# Get the directory of the current script
script_dir <- dirname(rprojroot::thisfile())


# Source the helpers.R from the same directory
source(file.path(script_dir, "helpers.R"))

args <- commandArgs(trailingOnly = TRUE)

# Check if a file path is provided
if (length(args) == 0) {
  stop("Please provide a file path to the summary file.")
}

# Read the provided file path from the command line
file_path <- args[1] #summary_all.tsv 
to_save <- args[2] #"figure.pdf"

# Read the TSV file using the provided file path
summary_all <- read_tsv(file_path)

##################################################
# FIGURE 3a
##################################################

# Add the new column for method types
column_info <- bind_rows(
  tribble(
    ~id, ~id_color, ~name, ~group, ~geom, ~palette, ~options,
    "method_name", NA_character_, "Name", "method", "text", NA_character_, list(width = 6, hjust = 0),
    "method_type", NA_character_, "Modality", "method", "text", NA_character_, list(width = 2, hjust = 0),
    "overall_score", "overall_score", "Score", "overall", "bar", "overall", list(width = 4),
    "R1 (all)", "R1 (all)", "R1 (all)", "metric_1", "funkyrect",  "metric_1", list(width = 2),
    "R1 (grn)", "R1 (grn)", "R1 (grn)", "metric_1", "funkyrect",  "metric_1", list(width = 2),
    "R2 (min)", "R2 (min)", "R2 (min)", "metric_2", "funkyrect",  "metric_2", list(width = 2),
    "R2 (med)", "R2 (med)", "R2 (med)", "metric_2", "funkyrect",  "metric_2", list(width = 2),
    "R2 (max)", "R2 (max)", "R2 (max)", "metric_2", "funkyrect",  "metric_2", list(width = 2),      
    "OPSCA", "OPSCA", "OPSCA", "dataset", "funkyrect", "dataset", list(width = 2),
    "Adamson", "Adamson", "Adamson", "dataset", "funkyrect", "dataset", list(width = 2),
    "Nakatake", "Nakatake", "Nakatake", "dataset", "funkyrect", "dataset", list(width = 2),
    "Norman", "Norman", "Norman", "dataset", "funkyrect", "dataset", list(width = 2),
    "Replogle", "Replogle", "Replogle", "dataset", "funkyrect", "dataset", list(width = 2),
  ),
  tribble(
    ~id, ~name, ~geom,
    "memory_log", "Peak memory (GB)", "rect",
    "memory_str", "", "text",
    "duration_log", "Duration (hour)", "rect",
    "duration_str", "", "text",
    "complexity_log", "Complexity", "rect",
    "Complexity", "", "text"
  ) %>% mutate(
    group = "resources",
    palette = ifelse(geom == "text", NA_character_, "resources"),
    options = map(geom, function(geom) {
      if (geom == "text") {
        list(overlay = TRUE, size = 2.5)
      } else {
        list(width = 2)
      }
    })
  )
)

# Define the method type mapping
method_type_mapping <- tribble(
  ~method_name, ~method_type,
  "GRNBoost2", "S",
  "GENIE3", "S",
  "PPCOR", "S",
  "Scenic", "S",
  "Portia", "S",
  "scGLUE", "M",
  "CellOracle", "M",
  "Scenic+", "M",
  "FigR", "M",
  "GRaNIE", "M",
  "Positive Control", "C",
  "Negative Control", "C",
  "Baseline Correlation", "C"
)


# Include the method types in the summary_all DataFrame
summary_all <- summary_all %>%
  left_join(method_type_mapping, by = "method_name")
print(summary_all)
# Update column groups to include the new "Type" column
column_groups <- tribble(
  ~group, ~palette, ~level1,
  "method", NA_character_, "",
  "overall", "overall", "Overall",
  "metric_1", "metric_1", "Regression 1",
  "metric_2", "metric_2", "Regression 2",
  "resources", "resources", "Resources", 
  "dataset", "dataset", "Datasets"
)

# Add palettes for the new column
palettes <- list(
  overall = "Greys",
  metric_1 = "Blues",
  metric_2 = "Reds",
  resources = "YlOrBr"
)

# Update legends if necessary
legends <- list(
  list(
    title = "Rank",
    palette = "overall",
    geom = "rect",
    labels = c("", "", "worst", "", "", "", "best", ""),
    label_hjust = rep(.5, 8),
    size = c(0, 0, 1, 1, 1, 1, 1, 0)
  ),
  list(
    title = "Scaled score",
    palette = "overall",
    geom = "funkyrect",
    labels = c("", "0", "", "", "", "0.4", "", "0.6", "", "0.8", "", "1"),
    size = c(0, seq(0, 1, by = .1)),
    label_hjust = rep(.5, 12)
  ),
  list(palette = "metric_1", enabled = FALSE),
  list(palette = "metric_2", enabled = FALSE),
  list(
    title = "Resources",
    palette = "resources",
    geom = "rect",
    labels = c("min", "", "", "", "max"),
    label_hjust = c(0, .5, .5, .5, 1),
    color = colorRampPalette(rev(funkyheatmap:::default_palettes$numerical$YlOrBr))(5),
    size = c(1, 1, 1, 1, 1)
  )
)

# Create the funkyheatmap
g3 <- funky_heatmap(
  data = summary_all,
  column_info = column_info %>% filter(id %in% colnames(summary_all)),
  column_groups = column_groups,
  palettes = palettes,
  position_args = position_arguments(
    expand_xmax = 2,
    col_annot_offset = max(str_length(column_info$name)) / 5
  ),
  add_abc = TRUE,
  scale_column = FALSE,
  legends = legends
) +
  theme(
    text = element_text(family = "Liberation Sans", size = 10), # Change font family and size
    plot.title = element_text(family = "Liberation Sans", face = "bold", size = 12), # Title font customization
    axis.text = element_text(size = 10),  # Axis text customization
    legend.text = element_text(size = 10) # Legend text customization
  )
ggsave(
  paste0(to_save,'.pdf'),
  g3,
  width = g3$width +2,
  height = g3$height + 2
)

ggsave(
  paste0(to_save,'.png'),
  g3,
  width = g3$width +2,
  height = g3$height + 2,
  dpi=300
)
