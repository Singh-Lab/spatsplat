#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
force <- "--force" %in% args

load_project_env <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[[1]])
    script_path <- normalizePath(script_path, mustWork = FALSE)
    project_root <- dirname(dirname(script_path))
  } else {
    project_root <- getwd()
  }

  env_path <- file.path(project_root, ".env")
  if (!file.exists(env_path)) {
    return(invisible(FALSE))
  }

  env_lines <- readLines(env_path, warn = FALSE)
  env_lines <- trimws(env_lines)
  env_lines <- env_lines[nzchar(env_lines) & !startsWith(env_lines, "#")]
  for (line in env_lines) {
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    if (length(parts) < 2) {
      next
    }
    key <- trimws(parts[[1]])
    value <- trimws(paste(parts[-1], collapse = "="))
    value <- sub('^"(.*)"$', "\\1", value)
    value <- sub("^'(.*)'$", "\\1", value)
    if (Sys.getenv(key, unset = "") == "") {
      do.call(Sys.setenv, stats::setNames(list(value), key))
    }
  }
  invisible(TRUE)
}

load_project_env()

data_dir <- Sys.getenv("SPATSPLAT_DATA_DIR", unset = "")
if (data_dir == "") {
  stop("SPATSPLAT_DATA_DIR is not set")
}

dataset_name <- Sys.getenv("PDAC_METS_VISIUM_DATASET_NAME", unset = "pdac_mets_visium_gse272362")
raw_dir <- file.path(data_dir, "raw", dataset_name)
processed_dir <- file.path(data_dir, "processed", dataset_name)
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

rds_path <- file.path(raw_dir, "PDAC_Updated.rds")
if (!file.exists(rds_path)) {
  stop(sprintf("Could not find RDS file: %s", rds_path))
}

counts_mtx_path <- file.path(processed_dir, "spatial_counts.mtx")
counts_genes_path <- file.path(processed_dir, "spatial_counts_genes.tsv.gz")
counts_barcodes_path <- file.path(processed_dir, "spatial_counts_barcodes.tsv.gz")
spot_metadata_path <- file.path(processed_dir, "spot_metadata.csv.gz")
image_scale_factors_path <- file.path(processed_dir, "image_scale_factors.csv.gz")
export_summary_path <- file.path(processed_dir, "export_summary.txt")

output_paths <- c(
  counts_mtx_path,
  counts_genes_path,
  counts_barcodes_path,
  spot_metadata_path,
  image_scale_factors_path,
  export_summary_path
)

if (!force && all(file.exists(output_paths))) {
  cat("All export files already exist. Use --force to overwrite.\n")
  quit(save = "no", status = 0)
}

cat(sprintf("Loading Seurat object from %s\n", rds_path))
obj <- readRDS(rds_path)

if (!inherits(obj, "Seurat")) {
  stop(sprintf("Expected a Seurat object, got: %s", paste(class(obj), collapse = ", ")))
}

get_counts_matrix <- function(seurat_obj) {
  if (!"Spatial" %in% Assays(seurat_obj)) {
    stop("Spatial assay not found in Seurat object")
  }

  counts <- tryCatch(
    GetAssayData(seurat_obj, assay = "Spatial", slot = "counts"),
    error = function(e) NULL
  )
  if (!is.null(counts)) {
    return(counts)
  }

  counts <- tryCatch(
    LayerData(seurat_obj[["Spatial"]], layer = "counts"),
    error = function(e) NULL
  )
  if (!is.null(counts)) {
    return(counts)
  }

  stop("Could not extract counts from Spatial assay")
}

counts <- get_counts_matrix(obj)
if (!inherits(counts, "sparseMatrix")) {
  counts <- as(counts, "dgCMatrix")
}

meta_df <- obj@meta.data
meta_df$cell_id <- rownames(meta_df)

image_names <- Images(obj)
if (length(image_names) == 0) {
  stop("No image objects found in Seurat object")
}

coords_list <- list()
scale_rows <- list()

for (img_name in image_names) {
  img_obj <- obj@images[[img_name]]
  coord_df <- img_obj@coordinates
  coord_df$cell_id <- rownames(coord_df)
  coord_df$image_name <- img_name
  coords_list[[img_name]] <- coord_df

  sf <- img_obj@scale.factors
  scale_rows[[img_name]] <- data.frame(
    image_name = img_name,
    spot = as.numeric(sf$spot),
    fiducial = as.numeric(sf$fiducial),
    hires = as.numeric(sf$hires),
    lowres = as.numeric(sf$lowres),
    stringsAsFactors = FALSE
  )
}

coords_df <- do.call(rbind, coords_list)
rownames(coords_df) <- NULL

scale_df <- do.call(rbind, scale_rows)
rownames(scale_df) <- NULL

spot_df <- merge(meta_df, coords_df, by = "cell_id", all.x = TRUE, sort = FALSE)

missing_coords <- sum(is.na(spot_df$image_name))
if (missing_coords > 0) {
  warning(sprintf("%d spots are missing image/coordinate records", missing_coords))
}

barcodes <- colnames(counts)
genes <- rownames(counts)

match_idx <- match(barcodes, spot_df$cell_id)
if (any(is.na(match_idx))) {
  missing_n <- sum(is.na(match_idx))
  stop(sprintf("Failed to match %d count-matrix barcodes to metadata rows", missing_n))
}
spot_df <- spot_df[match_idx, , drop = FALSE]

cat(sprintf("Writing counts matrix to %s\n", counts_mtx_path))
Matrix::writeMM(counts, file = counts_mtx_path)

cat(sprintf("Writing gene list to %s\n", counts_genes_path))
gz_genes <- gzfile(counts_genes_path, open = "wt")
writeLines(genes, gz_genes)
close(gz_genes)

cat(sprintf("Writing barcode list to %s\n", counts_barcodes_path))
gz_barcodes <- gzfile(counts_barcodes_path, open = "wt")
writeLines(barcodes, gz_barcodes)
close(gz_barcodes)

cat(sprintf("Writing spot metadata to %s\n", spot_metadata_path))
gz_spot_meta <- gzfile(spot_metadata_path, open = "wt")
write.csv(spot_df, gz_spot_meta, row.names = FALSE)
close(gz_spot_meta)

cat(sprintf("Writing image scale factors to %s\n", image_scale_factors_path))
gz_scale <- gzfile(image_scale_factors_path, open = "wt")
write.csv(scale_df, gz_scale, row.names = FALSE)
close(gz_scale)

summary_lines <- c(
  sprintf("dataset_name=%s", dataset_name),
  sprintf("raw_rds=%s", rds_path),
  sprintf("n_spots=%d", ncol(counts)),
  sprintf("n_genes=%d", nrow(counts)),
  sprintf("n_images=%d", length(image_names)),
  sprintf("default_assay=%s", DefaultAssay(obj)),
  sprintf("assays=%s", paste(Assays(obj), collapse = ",")),
  sprintf("meta_cols=%s", paste(colnames(meta_df), collapse = ",")),
  sprintf("images=%s", paste(image_names, collapse = ","))
)
writeLines(summary_lines, export_summary_path)

cat("Export complete.\n")
cat(sprintf("Processed files written to %s\n", processed_dir))
