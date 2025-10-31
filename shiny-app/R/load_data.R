# Data loading helpers ------------------------------------------------------------------

check_required_packages()

library(dplyr)

ensure_columns <- function(data, required, dataset) {
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop(sprintf("Dataset %s is missing required columns: %s", dataset, paste(missing, collapse = ", ")))
  }
  invisible(data)
}

read_parquet_tbl <- function(path, col_select = NULL) {
  if (!file.exists(path)) {
    stop(sprintf("Precomputed dataset not found: %s", path))
  }
  if (is.null(col_select)) {
    arrow::read_parquet(path) |> tibble::as_tibble()
  } else {
    arrow::read_parquet(path, col_select = tidyselect::any_of(col_select)) |> tibble::as_tibble()
  }
}

# Overview counts -----------------------------------------------------------------------

.overview_counts <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("overview_counts.parquet"),
    col_select = c("term", "n_bills", "n_committees", "n_legislators", "n_donors", "n_lobby_firms")
  ) |> ensure_columns(
    c("term", "n_bills", "n_committees", "n_legislators", "n_donors", "n_lobby_firms"),
    "overview_counts"
  )
})

get_overview_counts <- function(term) {
  .overview_counts() |> filter(.data$term == term)
}

get_available_terms <- memoise::memoise(function() {
  .overview_counts() |> arrange(desc(.data$term)) |> pull(.data$term) |> unique()
})

# Geography ----------------------------------------------------------------------------

.county_shapes <- memoise::memoise(function() {
  if (!file.exists(COUNTY_GEOJSON)) {
    stop(sprintf("Missing county geometry file: %s", COUNTY_GEOJSON))
  }
  sf::st_read(COUNTY_GEOJSON, quiet = TRUE) |>
    dplyr::select("county_id", "county_name", "geometry")
})

get_county_shapes <- function() {
  .county_shapes()
}

.map_metrics <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("map_choropleth_by_term.parquet"),
    col_select = c("term", "county_id", "metric", "value")
  ) |> ensure_columns(c("term", "county_id", "metric", "value"), "map_choropleth_by_term")
})

get_map_metric <- function(term, metric_key) {
  .map_metrics() |>
    filter(.data$term == term, .data$metric == metric_key)
}

.county_financials <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("county_funding_by_term.parquet"),
    col_select = c("term", "county_id", "total_donations", "total_lobbying", "total_received")
  ) |> ensure_columns(
    c("term", "county_id", "total_donations", "total_lobbying", "total_received"),
    "county_funding_by_term"
  )
})

get_county_financials <- function(county_id) {
  .county_financials() |>
    filter(.data$county_id == county_id)
}

.geo_spread_entities <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("geo_spread_entities.parquet"),
    col_select = c("term", "entity_type", "entity_name", "total", "county_entropy", "n_counties_active")
  ) |> ensure_columns(
    c("term", "entity_type", "entity_name", "total", "county_entropy", "n_counties_active"),
    "geo_spread_entities"
  )
})

get_geo_spread_entities <- function(term, entity_type = NULL) {
  df <- .geo_spread_entities() |> filter(.data$term == term)
  if (!is.null(entity_type)) {
    df <- df |> filter(.data$entity_type == entity_type)
  }
  df
}

# Topics ---------------------------------------------------------------------------------

.topic_division <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("topic_division_time.parquet"),
    col_select = c(
      "topic", "term", "mean_polarization", "party_line_share", "dem_yes_rate",
      "rep_yes_rate", "n_rollcalls", "controversiality", "momentum"
    )
  )
})

get_topic_summary <- function(min_rollcalls = 0, topic = NULL) {
  df <- .topic_division()
  if (min_rollcalls > 0) {
    df <- df |> filter(.data$n_rollcalls >= min_rollcalls)
  }
  if (!is.null(topic)) {
    df <- df |> filter(.data$topic == topic)
  }
  df
}

.topic_funding <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("topic_funding_by_term.parquet"),
    col_select = c("topic", "term", "total_donations", "total_lobbying", "total_received")
  )
})

get_topic_funding <- function(topic) {
  .topic_funding() |> filter(.data$topic == topic)
}

get_all_topic_funding <- function() {
  .topic_funding()
}

.topic_example_bills <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("bills_table.parquet"),
    col_select = c(
      "bill_ID", "bill_id_raw", "title", "topic", "term", "First_action",
      "longevity_days", "n_versions", "median_sim", "outcome", "vote_signal",
      "route_key"
    )
  )
})

get_topic_example_bills <- function(topic, term = NULL) {
  df <- .topic_example_bills() |> filter(.data$topic == topic)
  if (!is.null(term)) {
    df <- df |> filter(.data$term == term)
  }
  df
}

# People ---------------------------------------------------------------------------------

.committee_metrics <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("committee_gatekeeping.parquet"),
    col_select = c("committee", "term", "gatekeeping")
  ) |> left_join(
    read_parquet_tbl(
      precomputed_path("committee_workload_median.parquet"),
      col_select = c("committee", "term", "median_weekly_bills")
    ),
    by = c("committee", "term")
  )
})

get_committee_metrics <- function(term) {
  .committee_metrics() |> filter(.data$term == term)
}

.actor_overall <- memoise::memoise(function() {
  path <- precomputed_path("actor_overall.parquet")
  if (!file.exists(path)) return(tibble::tibble())
  read_parquet_tbl(path)
})

get_actor_overall <- function(term) {
  df <- .actor_overall()
  if (!nrow(df)) return(df)
  if ("term" %in% names(df)) {
    df <- df |> filter(.data$term == term)
  }
  df
}

.actor_topic <- memoise::memoise(function() {
  path <- precomputed_path("actor_topic.parquet")
  if (!file.exists(path)) return(tibble::tibble())
  read_parquet_tbl(path)
})

get_actor_topic <- function(term, topic = NULL) {
  df <- .actor_topic()
  if (!nrow(df)) return(df)
  if ("term" %in% names(df)) {
    df <- df |> filter(.data$term == term)
  }
  if (!is.null(topic) && topic != TOPIC_ALL_LABEL) {
    df <- df |> filter(.data$topic == topic)
  }
  df
}

# Donors ---------------------------------------------------------------------------------

.entity_stance <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("entity_stance_by_term.parquet"),
    col_select = c("term", "entity_type", "entity_name", "stance_score", "lean", "total")
  )
})

get_entity_stance <- function(term, entity_type = NULL) {
  df <- .entity_stance() |> filter(.data$term == term)
  if (!is.null(entity_type)) {
    df <- df |> filter(.data$entity_type == entity_type)
  }
  df
}

.donor_topic <- memoise::memoise(function() {
  path <- precomputed_path("donor_topic_by_term.parquet")
  if (!file.exists(path)) return(tibble::tibble())
  read_parquet_tbl(path)
})

get_donor_topic_allocations <- function(term, entity_name) {
  df <- .donor_topic()
  if (!nrow(df)) return(df)
  name_col <- intersect(c("entity_name", "ExpenderName"), names(df))[1]
  if (is.na(name_col)) {
    return(tibble::tibble())
  }
  df |> filter(.data$term == term, .data[[name_col]] == entity_name)
}

# Bills ----------------------------------------------------------------------------------

.pipeline_funnel <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("pipeline_stage_funnel.parquet"),
    col_select = c("term", "topic", "from", "to", "entered", "advanced", "pass_rate", "median_days")
  )
})

get_pipeline_funnel <- function(term, topic = NULL) {
  df <- .pipeline_funnel() |> filter(.data$term == term)
  if (!is.null(topic) && topic != TOPIC_ALL_LABEL) {
    df <- df |> filter(.data$topic == topic)
  }
  df
}

.route_archetypes <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("route_archetypes.parquet"),
    col_select = c("term", "topic", "route_key", "n", "pass_rate")
  )
})

get_route_archetypes <- function(term, topic = NULL) {
  df <- .route_archetypes() |> filter(.data$term == term)
  if (!is.null(topic) && topic != TOPIC_ALL_LABEL) {
    df <- df |> filter(.data$topic == topic)
  }
  df
}

.survival_curves <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("survival_curves.parquet"),
    col_select = c("term", "topic", "t", "survival")
  )
})

get_survival_curves <- function(term, topic = NULL) {
  df <- .survival_curves() |> filter(.data$term == term)
  if (!is.null(topic) && topic != TOPIC_ALL_LABEL) {
    df <- df |> filter(.data$topic == topic)
  }
  df
}

.bills_table <- memoise::memoise(function() {
  read_parquet_tbl(
    precomputed_path("bills_table.parquet"),
    col_select = c(
      "bill_ID", "bill_id_raw", "title", "topic", "term", "First_action",
      "longevity_days", "n_versions", "median_sim", "outcome", "vote_signal"
    )
  )
})

get_bills_table <- function(term, topic = NULL, outcome = NULL, route_key = NULL) {
  df <- .bills_table() |> filter(.data$term == term)
  if (!is.null(topic) && topic != TOPIC_ALL_LABEL) {
    df <- df |> filter(.data$topic == topic)
  }
  if (!is.null(outcome) && outcome != "All") {
    df <- df |> filter(.data$outcome == outcome)
  }
  if (!is.null(route_key) && nchar(route_key) && "route_key" %in% names(df)) {
    df <- df |> filter(.data$route_key == route_key)
  }
  df
}

.per_bill_model <- memoise::memoise(function() {
  path <- precomputed_path("per_bill.parquet")
  if (!file.exists(path)) return(tibble::tibble())
  read_parquet_tbl(path)
})

get_per_bill_model <- function(bill_id) {
  df <- .per_bill_model()
  if (!nrow(df)) return(df)
  df |> filter(.data$bill_ID == bill_id)
}
