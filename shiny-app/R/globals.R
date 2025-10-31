# Global constants and shared lists -----------------------------------------------------

APP_TITLE <- "California Legislative Insights"
DATA_DIR <- "data/precomputed"
GEO_DIR <- file.path(DATA_DIR, "geo")
TOPIC_ALL_LABEL <- "All topics"
DEFAULT_TERM <- NULL
SEARCH_DEBOUNCE_MS <- 300

MAP_METRIC_LABELS <- c(
  Donations = "total_donations",
  Lobbying = "total_lobbying",
  Received = "total_received"
)

COUNTY_GEOJSON <- file.path(DATA_DIR, "counties.geojson")

# Helper to build dataset paths ---------------------------------------------------------

precomputed_path <- function(filename) {
  file.path(DATA_DIR, filename)
}

# ensure required packages are available when app starts --------------------------------

required_packages <- c(
  "shiny", "bslib", "arrow", "DT", "leaflet", "plotly",
  "sf", "dplyr", "tidyr", "purrr", "stringr", "readr", "jsonlite", "memoise",
  "scales", "tidyselect"
)

#' Validate that all required packages are installed.
check_required_packages <- function() {
  missing <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      sprintf("Missing required packages: %s", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
  invisible(TRUE)
}
