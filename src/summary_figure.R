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


# create column info
column_info <-
  bind_rows(
    tribble(
      ~id, ~id_color, ~name, ~group, ~geom, ~palette, ~options,
      "method_name", NA_character_, "Name", "method", "text", NA_character_, list(width = 10, hjust = 0),
      "overall_score", "overall_score", "Score", "overall", "bar", "overall", list(width = 4),
      "S1", "S1", "S1", "metric_1", "funkyrect",  "metric_1", list(width = 2),
      "S2", "S2", "S2", "metric_1", "funkyrect",  "metric_1", list(width = 2),
      "static-theta-0.0", "static-theta-0.0", "Theta (min)", "metric_2", "funkyrect",  "metric_2", list(width = 2),
      "static-theta-0.5", "static-theta-0.5", "Theta (median)", "metric_2", "funkyrect",  "metric_2", list(width = 2),      
    ),
    tribble(
      ~id, ~name, ~geom,
      "memory_log", "Peak memory (GB)", "rect",
      "memory_str", "", "text",
      "duration_log", "Duration (hour)", "rect",
      "duration_str", "", "text"
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
print(column_info)
# create column groups
column_groups <- tribble(
  ~group, ~palette, ~level1,
  "method", NA_character_, "",
  "overall", "overall", "Overall",
  "metric_1", "metric_1", "Regression 1",
  "metric_2", "metric_2", "Regression 2",
  "resources", "resources", "Resources"
)

# create palettes
palettes <- list(
  overall = "Greys",
  metric_1 = "Blues",
  metric_2 = "Reds",
  resources = "YlOrBr"
)

# create palettes
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


# create funkyheatmap
g3 <- funky_heatmap(
  data = summary_all,
  column_info = column_info %>% filter(id %in% colnames(summary_all)),
  column_groups = column_groups,
  palettes = palettes,
  position_args = position_arguments(
    # determine xmax expand heuristically
    expand_xmax = 2,
    # determine offset heuristically
    col_annot_offset = max(str_length(column_info$name)) / 5
  ),
  add_abc = TRUE,
  scale_column = FALSE,
  legends = legends
)
ggsave(
  to_save,
  g3,
  width = g3$width +2,
  height = g3$height + 2
)
