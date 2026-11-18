# Install the comparator versions used by the distributed assessment.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop("Usage: Rscript install_r_dependencies.R LIBRARY_DIRECTORY")
dir.create(args[1], recursive=TRUE, showWarnings=FALSE)
library_dir <- normalizePath(args[1], mustWork=TRUE)
.libPaths(c(library_dir, .libPaths()))
options(repos=c(CRAN="https://cloud.r-project.org"))
versions <- c(ncvreg="3.16.0", ashr="2.2-63", jsonlite="2.0.0")
for (package in names(versions)) {
  available <- requireNamespace(package, quietly=TRUE)
  if (!available || packageVersion(package) != package_version(versions[[package]])) {
    if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", lib=library_dir)
    remotes::install_version(package, version=versions[[package]], lib=library_dir,
                             dependencies=NA, upgrade="never")
  }
  stopifnot(packageVersion(package) == package_version(versions[[package]]))
  cat(package, as.character(packageVersion(package)), "\n")
}
cat("R comparator dependencies are available in", library_dir, "\n")
