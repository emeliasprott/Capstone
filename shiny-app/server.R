library(shiny)
library(bslib)

APP_TITLE <- "California Legislative Insights"
DATA_DIR <- "data/precomputed"
GEO_DIR <- file.path(DATA_DIR, "geo")
TOPIC_ALL_LABEL <- "All topics"
MAP_METRIC_LABELS <- c(
  Donations = "total_donations",
  Lobbying = "total_lobbying",
  Received = "total_received"
)
COUNTY_GEOJSON <- file.path(DATA_DIR, "counties.geojson")

required_packages <- c(
  "shiny", "bslib", "arrow", "DT", "leaflet", "plotly",
  "sf", "dplyr", "tidyr", "purrr", "stringr", "readr", "jsonlite", "memoise",
  "scales", "tidyselect"
)

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

check_required_packages()

library(dplyr)

fmt_money <- function(x, digits = 0) {
  if (is.null(x)) {
    return(rep("—", length.out = length(x)))
  }
  res <- scales::dollar(x, accuracy = 1 / (10^digits), trim = TRUE)
  res[is.na(x)] <- "—"
  res
}

fmt_pct0 <- function(x) {
  res <- scales::percent(x, accuracy = 1)
  res[is.na(x)] <- "—"
  res
}

fmt_pct1 <- function(x) {
  res <- scales::percent(x, accuracy = 0.1)
  res[is.na(x)] <- "—"
  res
}

fmt_date <- function(x) {
  if (inherits(x, "Date")) {
    format(x, "%b %d, %Y")
  } else {
    x
  }
}

pal_sequential <- function(n = 7L) {
  grDevices::colorRampPalette(c("#e8f0fe", "#2B4C7E"))(n)
}

pal_diverging <- function(n = 7L) {
  grDevices::colorRampPalette(c("#81312f", "#f5f2f0", "#0F766E"))(n)
}

rank_and_percentile <- function(x, decreasing = TRUE) {
  if (!length(x)) {
    return(list(rank = integer(), percentile = numeric()))
  }
  ord <- if (decreasing) -x else x
  ranks <- rank(ord, ties.method = "min")
  pct <- ranks / max(ranks)
  list(rank = ranks, percentile = pct)
}

safe_pull <- function(data, column, default = NA) {
  if (column %in% names(data)) {
    data[[column]]
  } else {
    default
  }
}

sanitize_id <- function(x) {
  stringr::str_replace_all(tolower(x), "[^a-z0-9]+", "-")
}

null_default <- function(x, default) {
  if (is.null(x) || !length(x)) default else x
}

precomputed_path <- function(filename) {
  file.path(DATA_DIR, filename)
}

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

parse_query <- function(session) {
  shiny::parseQueryString(isolate(session$clientData$url_search))
}

encode_query <- function(params) {
  if (!length(params)) return("")
  pairs <- purrr::imap_chr(params, function(value, key) {
    paste0(utils::URLencode(key, reserved = TRUE), "=", utils::URLencode(as.character(value), reserved = TRUE))
  })
  paste0("?", paste(pairs, collapse = "&"))
}

update_query <- function(session, params, mode = c("replace", "push")) {
  mode <- match.arg(mode)
  current <- parse_query(session)
  merged <- modifyList(current, params)
  merged <- merged[!vapply(merged, is.null, logical(1))]
  shiny::updateQueryString(encode_query(merged), mode = mode, session = session)
}

get_query_value <- function(session, key, default = NULL) {
  query <- parse_query(session)
  if (!is.null(query[[key]])) query[[key]] else default
}

mod_header_server <- function(id, terms, topics, default_term, default_topic, default_search, on_state_change) {
  moduleServer(id, function(input, output, session) {
    term_values <- null_default(terms, "2023-2024")
    topic_values <- unique(c(TOPIC_ALL_LABEL, null_default(topics, character())))

    term_state <- reactiveVal(NULL)
    topic_state <- reactiveVal(NULL)
    search_state <- reactiveVal(NULL)

    observeEvent(TRUE, {
      choices <- setNames(term_values, term_values)
      selected <- default_term
      if (is.null(selected) || !selected %in% term_values) {
        selected <- term_values[[1]]
      }
      updateSelectInput(session, "term", choices = choices, selected = selected)
      term_state(selected)
    }, once = TRUE)

    observeEvent(TRUE, {
      topic_choices <- unique(c(TOPIC_ALL_LABEL, sort(topic_values)))
      selected <- default_topic
      if (is.null(selected) || !selected %in% topic_choices) {
        selected <- TOPIC_ALL_LABEL
      }
      updateSelectizeInput(session, "topic", choices = topic_choices, selected = selected, server = TRUE)
      topic_state(selected)
    }, once = TRUE)

    observeEvent(TRUE, {
      value <- null_default(default_search, "")
      updateTextInput(session, "search", value = value)
      search_state(value)
    }, once = TRUE)

    observeEvent(input$term, {
      term_state(input$term)
      on_state_change()
    }, ignoreNULL = FALSE)

    observeEvent(input$topic, {
      topic_state(input$topic)
      on_state_change()
    }, ignoreNULL = FALSE)

    observeEvent(input$search, {
      search_state(input$search)
    }, ignoreNULL = FALSE)

    observeEvent(input$copy_link, {
      session$sendCustomMessage("copy-url", list(url = session$clientData$url_href))
    })

    list(
      term = reactive(term_state()),
      topic = reactive(topic_state()),
      search = reactive(search_state()),
      update_term = function(value) {
        if (!is.null(value) && !identical(value, term_state())) {
          updateSelectInput(session, "term", selected = value)
          term_state(value)
        }
      },
      update_topic = function(value) {
        if (!is.null(value) && !identical(value, topic_state())) {
          updateSelectizeInput(session, "topic", selected = value)
          topic_state(value)
        }
      },
      update_search = function(value) {
        if (!is.null(value) && !identical(value, search_state())) {
          updateTextInput(session, "search", value = value)
          search_state(value)
        }
      }
    )
  })
}

mod_overview_server <- function(id, term) {
  moduleServer(id, function(input, output, session) {
    counts <- reactive({
      req(term())
      df <- get_overview_counts(term())
      if (!nrow(df)) {
        tibble::tibble(
          n_bills = NA_integer_, n_committees = NA_integer_,
          n_legislators = NA_integer_, n_donors = NA_integer_, n_lobby_firms = NA_integer_
        )
      } else {
        df
      }
    }) |>
      bindCache(term())

    output$bills_value <- renderUI({
      value <- counts()$n_bills[1]
      span(class = "metric-value", scales::comma(value))
    })

    output$committees_value <- renderUI({
      value <- counts()$n_committees[1]
      span(class = "metric-value", scales::comma(value))
    })

    output$legislators_value <- renderUI({
      value <- counts()$n_legislators[1]
      span(class = "metric-value", scales::comma(value))
    })

    output$money_value <- renderUI({
      donors <- counts()$n_donors[1]
      lobby <- counts()$n_lobby_firms[1]
      span(class = "metric-value", sprintf("%s donors / %s lobby", scales::comma(donors), scales::comma(lobby)))
    })
  })
}

mod_map_server <- function(id, term, topic) {
  moduleServer(id, function(input, output, session) {
    shapes <- reactive(get_county_shapes())

    observe({
      req(term())
      entities <- get_geo_spread_entities(term())
      updateSelectizeInput(session, "entity_search", choices = sort(unique(entities$entity_name)), server = TRUE)
    })

    selected_county <- reactiveVal(NULL)

    metric_key <- reactive({
      MAP_METRIC_LABELS[[input$metric]]
    })

    parse_share_column <- function(x) {
      if (is.null(x) || length(x) == 0) return(NULL)
      entry <- x[[1]]
      if (is.null(entry)) return(NULL)
      if (is.character(entry)) {
        parsed <- tryCatch(jsonlite::fromJSON(entry), error = function(e) NULL)
      } else {
        parsed <- entry
      }
      if (is.null(parsed)) return(NULL)
      tibble::as_tibble(parsed)
    }

    entity_overlay <- reactive({
      req(term())
      name <- input$entity_search
      if (is.null(name) || !nzchar(name)) return(NULL)
      df <- get_geo_spread_entities(term())
      ent <- df |> dplyr::filter(.data$entity_name == name)
      if (!nrow(ent)) return(NULL)
      shares <- NULL
      if ("share_by_county" %in% names(ent)) {
        shares <- parse_share_column(ent$share_by_county)
      }
      list(row = ent, shares = shares)
    })

    map_data <- reactive({
      req(term(), metric_key())
      metric_df <- get_map_metric(term(), metric_key())
      geo <- shapes()
      joined <- dplyr::left_join(geo, metric_df, by = "county_id")
      stats <- rank_and_percentile(joined$value)
      joined$rank <- stats$rank
      joined$percentile <- stats$percentile
      overlay <- entity_overlay()
      if (!is.null(overlay) && !is.null(overlay$shares) && all(c("county_id", "share") %in% names(overlay$shares))) {
        joined <- dplyr::left_join(joined, overlay$shares, by = "county_id")
        joined$entity_share <- joined$share
      } else {
        joined$entity_share <- NA_real_
      }
      joined
    }) |>
      bindCache(term(), metric_key(), input$entity_search)

    output$county_map <- leaflet::renderLeaflet({
      leaflet::leaflet(options = leaflet::leafletOptions(preferCanvas = TRUE)) |>
        leaflet::addProviderTiles(leaflet::providers$CartoDB.Positron) |>
        leaflet::setView(lng = -119.5, lat = 37.25, zoom = 5.5)
    })

    observe({
      req(map_data())
      df <- map_data()
      pal <- leaflet::colorBin(palette = pal_sequential(7), domain = df$value, bins = 7, na.color = "#f5f5f5")
      proxy <- leaflet::leafletProxy("county_map", session = session, data = df)
      proxy |>
        leaflet::clearShapes() |>
        leaflet::addPolygons(
          fillColor = ~pal(value),
          weight = 0.6,
          color = "#ffffff",
          opacity = 0.8,
          fillOpacity = 0.8,
          layerId = ~county_id,
          highlightOptions = leaflet::highlightOptions(weight = 2, color = "#2B4C7E", bringToFront = TRUE),
          label = ~sprintf(
            "%s\n%s: %s\nRank: %s (%.0f%%)",
            county_name,
            input$metric,
            fmt_money(value, digits = 1),
            scales::comma(rank),
            percentile * 100
          ),
          labelOptions = leaflet::labelOptions(direction = "auto")
        )
      overlay <- entity_overlay()
      proxy |> leaflet::clearControls()
      if (!is.null(overlay) && nrow(overlay$row)) {
        ent <- overlay$row
        html <- htmltools::tags$div(
          class = "entity-overlay",
          htmltools::tags$strong(ent$entity_name),
          htmltools::tags$br(),
          sprintf("Total: %s", fmt_money(ent$total)),
          htmltools::tags$br(),
          sprintf("County spread (entropy): %.2f", safe_pull(ent, "county_entropy", NA_real_))
        )
        proxy |> leaflet::addControl(html = htmltools::as.tags(html), position = "topright")
      }
    })

    observeEvent(input$county_map_shape_click, {
      click <- input$county_map_shape_click
      if (!is.null(click$id)) {
        selected_county(click$id)
      }
    })

    county_timeseries <- reactive({
      req(selected_county())
      get_county_financials(selected_county())
    }) |>
      bindCache(selected_county())

    output$county_trends <- plotly::renderPlotly({
      df <- county_timeseries()
      validate(need(nrow(df), "Select a county from the map."))
      plotly::plot_ly(df, x = ~term, y = ~total_donations, type = "scatter", mode = "lines+markers", name = "Donations") |>
        plotly::add_trace(y = ~total_lobbying, name = "Lobbying") |>
        plotly::add_trace(y = ~total_received, name = "Received") |>
        plotly::layout(xaxis = list(title = "Term"), yaxis = list(title = "USD"), legend = list(orientation = "h"))
    })

    top_actors_data <- reactive({
      req(term())
      geo_df <- get_geo_spread_entities(term())
      county_id <- selected_county()
      overlay <- entity_overlay()
      if (!is.null(county_id) && !is.null(overlay) && !is.null(overlay$shares) && "county_id" %in% names(geo_df)) {
        shares <- overlay$shares
        if (all(c("county_id", "share") %in% names(shares))) {
          shares <- shares |> dplyr::filter(.data$county_id == county_id)
          geo_df <- geo_df |> dplyr::semi_join(shares, by = "county_id")
        }
      }
      geo_df |> dplyr::arrange(dplyr::desc(.data$total)) |> dplyr::mutate(rank = dplyr::row_number())
    }) |>
      bindCache(term(), selected_county())

    output$top_actors <- DT::renderDT({
      df <- top_actors_data()
      cols <- intersect(c("rank", "entity_name", "entity_type", "total", "county_share", "county_entropy"), names(df))
      if (!length(cols)) {
        return(DT::datatable(tibble::tibble(message = "No entity details available.")))
      }
      table <- df |> dplyr::select(dplyr::all_of(cols))
      DT::datatable(
        table,
        filter = "top",
        extensions = "Buttons",
        options = list(pageLength = 15, dom = "Bfrtip", buttons = c("copy", "csv")),
        caption = htmltools::tags$caption(
          style = "caption-side: top; text-align: left;",
          if (!is.null(selected_county())) {
            "Top entities for the selected county"
          } else {
            "Top entities statewide"
          }
        )
      )
    })
  })
}

mod_topics_server <- function(id, term, topic_filter) {
  moduleServer(id, function(input, output, session) {
    selected_topic <- reactiveVal(NULL)

    observeEvent(topic_filter(), {
      if (!is.null(topic_filter()) && topic_filter() != TOPIC_ALL_LABEL) {
        selected_topic(topic_filter())
      }
    })

    output$active_topic <- renderText({
      if (is.null(topic_filter()) || topic_filter() == TOPIC_ALL_LABEL) {
        "All topics"
      } else {
        sprintf("Focused on: %s", topic_filter())
      }
    })

    topics_data <- reactive({
      req(term())
      summary <- get_topic_summary(input$min_rollcalls)
      if (!is.null(topic_filter()) && topic_filter() != TOPIC_ALL_LABEL) {
        summary <- summary |> dplyr::filter(.data$topic == topic_filter())
      }
      current <- summary |> dplyr::filter(.data$term == term())
      funding <- get_all_topic_funding()
      funding_current <- funding |> dplyr::filter(.data$term == term())
      combined <- dplyr::left_join(current, funding_current, by = c("topic", "term"))
      if (!nrow(combined)) {
        return(tibble::tibble())
      }
      combined$spark <- purrr::map(combined$topic, ~ summary |> dplyr::filter(.data$topic == .x) |> dplyr::arrange(.data$term))
      combined$fund_trend <- purrr::map(combined$topic, ~ funding |> dplyr::filter(.data$topic == .x) |> dplyr::arrange(.data$term))
      combined$sort_key <- dplyr::case_when(
        input$sort_by == "Polarization" ~ combined$mean_polarization,
        input$sort_by == "Party-line share" ~ combined$party_line_share,
        input$sort_by == "Controversy" ~ safe_pull(combined, "controversiality", NA_real_),
        input$sort_by == "Momentum" ~ safe_pull(combined, "momentum", NA_real_),
        TRUE ~ combined$mean_polarization
      )
      combined |> dplyr::arrange(dplyr::desc(.data$sort_key))
    }) |>
      bindCache(term(), topic_filter(), input$min_rollcalls, input$sort_by)

    observeEvent(topics_data(), {
      df <- topics_data()
      if (!nrow(df)) return()
      current <- selected_topic()
      if (is.null(current) || !current %in% df$topic) {
        selected_topic(df$topic[[1]])
      }
    }, priority = -1)

    observeEvent(topics_data(), {
      df <- topics_data()
      if (!nrow(df)) return(NULL)
      purrr::walk(seq_len(nrow(df)), function(idx) {
        topic_name <- df$topic[[idx]]
        topic_id <- sanitize_id(topic_name)
        spark_id <- paste0("spark_", topic_id)
        fund_id <- paste0("fund_", topic_id)
        spark_data <- df$spark[[idx]]
        fund_data <- df$fund_trend[[idx]]
        output[[spark_id]] <- plotly::renderPlotly({
          validate(need(nrow(spark_data), ""))
          plotly::plot_ly(spark_data, x = ~term, y = ~mean_polarization, type = "scatter", mode = "lines", showlegend = FALSE) |>
            plotly::layout(margin = list(l = 10, r = 10, t = 10, b = 20), xaxis = list(title = NULL, showticklabels = FALSE), yaxis = list(title = NULL, showticklabels = FALSE))
        })
        output[[fund_id]] <- plotly::renderPlotly({
          validate(need(nrow(fund_data), ""))
          plotly::plot_ly(fund_data, x = ~term, y = ~total_received, type = "bar", showlegend = FALSE) |>
            plotly::layout(margin = list(l = 10, r = 10, t = 10, b = 20), xaxis = list(title = NULL, showticklabels = FALSE), yaxis = list(title = NULL, showticklabels = FALSE))
        })
        observeEvent(input[[paste0("select_", topic_id)]], {
          selected_topic(topic_name)
        }, ignoreNULL = TRUE)
      })
    })

    output$topic_cards <- renderUI({
      df <- topics_data()
      if (!nrow(df)) {
        return(bslib::card(bslib::card_body("No topics available for the current filters.")))
      }
      cards <- purrr::imap(df$topic, function(topic_name, idx) {
        topic_id <- sanitize_id(topic_name)
        spark_id <- session$ns(paste0("spark_", topic_id))
        fund_id <- session$ns(paste0("fund_", topic_id))
        action_id <- session$ns(paste0("select_", topic_id))
        stats <- df[idx, , drop = FALSE]
        funding_total <- safe_pull(stats, "total_received", NA_real_)
        card_body <- bslib::card_body(
          div(class = "topic-metrics",
              div(class = "sparkline", plotly::plotlyOutput(spark_id, height = 80)),
              div(class = "fundspark", plotly::plotlyOutput(fund_id, height = 80))),
          div(class = "mt-2", sprintf("Party-line votes: %s", fmt_pct0(stats$party_line_share))),
          div(class = "text-muted", sprintf("Roll calls: %s", scales::comma(stats$n_rollcalls))),
          if (!is.na(funding_total)) div(class = "text-muted", sprintf("Recent funding: %s", fmt_money(funding_total)))
        )
        bslib::card(
          class = sprintf("topic-card %s", if (identical(selected_topic(), topic_name)) "selected" else ""),
          bslib::card_header(
            shiny::actionLink(action_id, label = topic_name, class = "topic-card-link")
          ),
          card_body
        )
      })
      do.call(bslib::layout_columns, c(list(col_widths = rep(4, length(cards))), cards))
    })

    topic_detail_data <- reactive({
      topic_name <- selected_topic()
      if (is.null(topic_name)) return(NULL)
      df <- topics_data()
      if (!nrow(df) || !topic_name %in% df$topic) return(NULL)
      list(
        spark = df |> dplyr::filter(.data$topic == topic_name) |> dplyr::pull(.data$spark) |> purrr::pluck(1),
        funding = df |> dplyr::filter(.data$topic == topic_name) |> dplyr::pull(.data$fund_trend) |> purrr::pluck(1),
        bills = get_topic_example_bills(topic_name, term())
      )
    })

    output$topic_detail <- renderUI({
      detail <- topic_detail_data()
      if (is.null(detail)) return(NULL)
      ns <- session$ns
      spark_plot_id <- ns("detail_polarization")
      funding_plot_id <- ns("detail_funding")
      output$detail_polarization <- plotly::renderPlotly({
        df <- detail$spark
        plotly::plot_ly(df, x = ~term, y = ~mean_polarization, type = "scatter", mode = "lines+markers", name = "Polarization") |>
          plotly::layout(xaxis = list(title = "Term"), yaxis = list(title = "Mean polarization"))
      })
      output$detail_funding <- plotly::renderPlotly({
        df <- detail$funding
        plotly::plot_ly(df, x = ~term, y = ~total_received, type = "bar", name = "Funding") |>
          plotly::layout(xaxis = list(title = "Term"), yaxis = list(title = "Total received"))
      })
      output$topic_example_bills <- DT::renderDT({
        bills <- detail$bills
        if (!nrow(bills)) {
          return(DT::datatable(tibble::tibble(message = "No example bills for this topic.")))
        }
        if (!"bill_link" %in% names(bills) && "bill_ID" %in% names(bills)) {
          bills$bill_link <- sprintf('<a href="https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=%s" target="_blank">view bill</a>', bills$bill_ID)
        }
        display <- bills |> dplyr::select(-dplyr::any_of(c("bill_id_raw")))
        DT::datatable(
          display,
          filter = "top",
          extensions = "Buttons",
          escape = FALSE,
          options = list(pageLength = 10, dom = "Bfrtip", buttons = c("copy", "csv"))
        )
      })
      bslib::card(
        class = "mt-4",
        bslib::card_header(sprintf("Deep dive: %s", selected_topic())),
        bslib::card_body(
          bslib::layout_columns(
            bslib::card(plotly::plotlyOutput(spark_plot_id, height = 260)),
            bslib::card(plotly::plotlyOutput(funding_plot_id, height = 260))
          ),
          DT::DTOutput(ns("topic_example_bills"))
        )
      )
    })
  })
}

mod_people_server <- function(id, term, topic, search) {
  moduleServer(id, function(input, output, session) {
    selected_entity <- reactiveVal(NULL)

    people_data <- reactive({
      req(term())
      if (identical(input$people_tab, "Legislators")) {
        df <- get_actor_overall(term())
        if (nrow(df)) {
          if ("actor_type" %in% names(df)) {
            df <- df |> dplyr::filter(.data$actor_type == "Legislator")
          }
          if (!is.null(topic()) && topic() != TOPIC_ALL_LABEL) {
            topic_df <- get_actor_topic(term(), topic())
            if (nrow(topic_df)) {
              df <- dplyr::left_join(df, topic_df, by = c("actor_name", "actor_type"), suffix = c("_overall", "_topic"))
            }
          }
          if ("party" %in% names(df) && length(input$party_filter)) {
            df <- df |> dplyr::filter(.data$party %in% input$party_filter)
          }
          if ("chamber" %in% names(df) && length(input$chamber_filter)) {
            df <- df |> dplyr::filter(.data$chamber %in% input$chamber_filter)
          }
          if ("overall_influence" %in% names(df)) {
            df <- df |> dplyr::arrange(dplyr::desc(.data$overall_influence))
          }
        }
      } else {
        df <- get_committee_metrics(term())
        if (nrow(df) && length(input$chamber_filter) && "chamber" %in% names(df)) {
          df <- df |> dplyr::filter(.data$chamber %in% input$chamber_filter)
        }
      }
      df
    }) |>
      bindCache(term(), topic(), input$people_tab, input$party_filter, input$chamber_filter)

    summary_points <- reactive({
      df <- people_data()
      if (!nrow(df)) return(character())
      top_rows <- head(df, 3)
      purrr::map_chr(seq_len(nrow(top_rows)), function(i) {
        row <- top_rows[i, ]
        name_col <- intersect(c("actor_name", "committee", "name"), names(row))[1]
        metric_col <- intersect(c("overall_influence", "gatekeeping", "score", "median_weekly_bills"), names(row))[1]
        if (is.na(name_col) || is.na(metric_col)) return("Data unavailable for summary.")
        sprintf("%s ranks #%d with %s", row[[name_col]], i, scales::comma(row[[metric_col]]))
      })
    })

    output$summary_points <- htmltools::renderTags({
      items <- summary_points()
      if (!length(items)) {
        htmltools::tags$ul(htmltools::tags$li("No data available."))
      } else {
        htmltools::tags$ul(purrr::map(items, htmltools::tags$li))
      }
    })

    output$leader_table <- DT::renderDT({
      df <- people_data()
      if (!nrow(df)) {
        return(DT::datatable(tibble::tibble(message = "No results for these filters.")))
      }
      display_cols <- intersect(
        c("actor_name", "party", "chamber", "overall_influence", "top_topic", "events", "support", "committee", "gatekeeping", "median_weekly_bills"),
        names(df)
      )
      if (!length(display_cols)) display_cols <- names(df)
      DT::datatable(
        df |> dplyr::select(dplyr::all_of(display_cols)),
        filter = "top",
        selection = "single",
        extensions = "Buttons",
        options = list(pageLength = 15, dom = "Bfrtip", buttons = c("copy", "csv"))
      )
    })

    proxy <- DT::dataTableProxy("leader_table")
    observeEvent(search(), {
      DT::updateSearch(proxy, keywords = list(global = search()))
    })

    observeEvent(input$leader_table_rows_selected, {
      df <- people_data()
      idx <- input$leader_table_rows_selected
      if (length(idx) == 1 && idx <= nrow(df)) {
        selected_entity(df[idx, , drop = FALSE])
      }
    })

    output$trend_chart <- plotly::renderPlotly({
      row <- selected_entity()
      if (is.null(row)) {
        return(plotly::plotly_empty())
      }
      name_col <- intersect(c("actor_name", "committee", "name"), names(row))[1]
      if (is.na(name_col)) return(plotly::plotly_empty())
      history <- get_actor_topic(term(), topic())
      if (nrow(history) && name_col %in% names(history)) {
        history <- history |> dplyr::filter(.data[[name_col]] == row[[name_col]])
      }
      if (!nrow(history)) {
        return(plotly::plotly_empty())
      }
      history$score_val <- safe_pull(history, "score", NA_real_)
      plotly::plot_ly(history, x = ~term, y = ~score_val, type = "scatter", mode = "lines+markers", name = "Score") |>
        plotly::layout(xaxis = list(title = "Term"), yaxis = list(title = "Topic influence"))
    })

    output$drill_bills <- DT::renderDT({
      row <- selected_entity()
      if (is.null(row)) {
        return(DT::datatable(tibble::tibble(message = "Select a row above to see related bills.")))
      }
      focus_topic <- if (!is.null(topic()) && topic() != TOPIC_ALL_LABEL) topic() else NULL
      if (is.null(focus_topic)) {
        return(DT::datatable(tibble::tibble(message = "Set a topic filter to see related bills.")))
      }
      bills <- get_topic_example_bills(focus_topic, term())
      if (!nrow(bills)) {
        return(DT::datatable(tibble::tibble(message = "No related bills found.")))
      }
      DT::datatable(
        bills,
        filter = "top",
        extensions = "Buttons",
        options = list(pageLength = 10, dom = "Bfrtip", buttons = c("copy", "csv"))
      )
    })
  })
}

mod_donors_server <- function(id, term, topic, search) {
  moduleServer(id, function(input, output, session) {
    selected_entity <- reactiveVal(NULL)

    entity_data <- reactive({
      req(term())
      stance <- get_entity_stance(term())
      if (!nrow(stance)) return(tibble::tibble())
      spread <- get_geo_spread_entities(term())
      df <- dplyr::left_join(stance, spread, by = c("term", "entity_type", "entity_name", "total"))
      df
    }) |>
      bindCache(term())

    observe({
      df <- entity_data()
      if (!nrow(df)) return(NULL)
      updateSelectizeInput(session, "entity_search", choices = sort(unique(df$entity_name)), server = TRUE)
    })

    filtered_entities <- reactive({
      df <- entity_data()
      if (!nrow(df)) return(df)
      df <- df |> dplyr::filter(.data$entity_type == input$entity_type)
      if (!is.null(topic()) && topic() != TOPIC_ALL_LABEL && "topic" %in% names(df)) {
        df <- df |> dplyr::filter(.data$topic == topic())
      }
      df
    }) |>
      bindCache(term(), input$entity_type, topic())

    observeEvent(input$entity_search, {
      name <- input$entity_search
      if (!nzchar(name)) return()
      df <- filtered_entities()
      ent <- df |> dplyr::filter(.data$entity_name == name)
      if (nrow(ent)) selected_entity(ent[1, , drop = FALSE])
    })

    output$entity_scatter <- plotly::renderPlotly({
      df <- filtered_entities()
      if (!nrow(df)) return(plotly::plotly_empty())
      df$size <- sqrt(df$total)
      df$color <- safe_pull(df, "lean", "Neutral")
      plotly::plot_ly(
        df,
        x = ~1 / pmax(1, safe_pull(df, "n_counties_active", 1)),
        y = ~stance_score,
        type = "scatter",
        mode = "markers",
        text = ~paste0(entity_name, "\nTotal: ", fmt_money(total), "\nEntropy: ", round(safe_pull(df, "county_entropy", NA_real_), 2)),
        hoverinfo = "text",
        color = ~color,
        size = ~size,
        sizes = c(10, 60),
        customdata = ~entity_name,
        source = session$ns("entity_scatter")
      ) |>
        plotly::layout(
          xaxis = list(title = "Geographic concentration (left = concentrated)", showticklabels = FALSE),
          yaxis = list(title = "Stance score"),
          showlegend = FALSE
        )
    })

    observeEvent(plotly::event_data("plotly_click", source = session$ns("entity_scatter")), {
      click <- plotly::event_data("plotly_click", source = session$ns("entity_scatter"))
      if (is.null(click)) return()
      name <- click$customdata
      df <- filtered_entities()
      if (!is.null(name) && name %in% df$entity_name) {
        selected_entity(df |> dplyr::filter(.data$entity_name == name) |> dplyr::slice_head(n = 1))
      }
    })

    output$entity_table <- DT::renderDT({
      df <- filtered_entities()
      if (!nrow(df)) {
        return(DT::datatable(tibble::tibble(message = "No entities available for this view.")))
      }
      display <- df |> dplyr::select(dplyr::any_of(c("entity_name", "stance_score", "lean", "total", "county_entropy", "n_counties_active")))
      DT::datatable(
        display,
        filter = "top",
        selection = "single",
        extensions = "Buttons",
        options = list(pageLength = 15, dom = "Bfrtip", buttons = c("copy", "csv"))
      )
    })

    proxy <- DT::dataTableProxy("entity_table")
    observeEvent(search(), {
      DT::updateSearch(proxy, keywords = list(global = search()))
    })

    observeEvent(input$entity_table_rows_selected, {
      df <- filtered_entities()
      idx <- input$entity_table_rows_selected
      if (length(idx) == 1 && idx <= nrow(df)) {
        selected_entity(df[idx, , drop = FALSE])
      }
    })

    parse_share_column <- function(entity_row) {
      if (!"share_by_county" %in% names(entity_row)) return(NULL)
      val <- entity_row$share_by_county[[1]]
      if (is.null(val)) return(NULL)
      if (is.character(val)) {
        val <- tryCatch(jsonlite::fromJSON(val), error = function(e) NULL)
      }
      if (is.null(val)) return(NULL)
      tibble::as_tibble(val)
    }

    output$entity_map <- leaflet::renderLeaflet({
      ent <- selected_entity()
      if (is.null(ent)) {
        return(leaflet::leaflet() |> leaflet::addProviderTiles(leaflet::providers$CartoDB.Positron))
      }
      shares <- parse_share_column(ent)
      if (is.null(shares) || !all(c("county_id", "share") %in% names(shares))) {
        return(leaflet::leaflet() |> leaflet::addProviderTiles(leaflet::providers$CartoDB.Positron) |>
                 leaflet::addControl("County-level allocations not available.", position = "topright"))
      }
      shapes <- get_county_shapes()
      df <- dplyr::left_join(shapes, shares, by = "county_id")
      pal <- leaflet::colorBin(pal_sequential(7), domain = df$share, bins = 7, na.color = "#f5f5f5")
      leaflet::leaflet(df) |>
        leaflet::addProviderTiles(leaflet::providers$CartoDB.Positron) |>
        leaflet::addPolygons(
          fillColor = ~pal(share),
          color = "#ffffff",
          weight = 0.6,
          opacity = 0.8,
          fillOpacity = 0.8,
          label = ~sprintf("%s: %.1f%%", county_name, share * 100)
        )
    })

    output$entity_topics <- plotly::renderPlotly({
      ent <- selected_entity()
      if (is.null(ent)) return(plotly::plotly_empty())
      df <- get_donor_topic_allocations(term(), ent$entity_name)
      if (!nrow(df)) {
        return(plotly::plotly_empty())
      }
      df$value <- safe_pull(df, "donations_allocated", safe_pull(df, "total", NA_real_))
      plotly::plot_ly(df, x = ~topic, y = ~value, type = "bar") |>
        plotly::layout(xaxis = list(title = "Topic"), yaxis = list(title = "Allocated"))
    })
  })
}

mod_bills_server <- function(id, term, topic, search) {
  moduleServer(id, function(input, output, session) {
    selected_route <- reactiveVal(NULL)
    selected_bill <- reactiveVal(NULL)

    funnel_data <- reactive({
      req(term())
      get_pipeline_funnel(term(), topic())
    }) |>
      bindCache(term(), topic())

    route_data <- reactive({
      req(term())
      get_route_archetypes(term(), topic())
    }) |>
      bindCache(term(), topic())

    observe({
      routes <- route_data()
      choices <- c("All", sort(unique(routes$route_key)))
      updateSelectizeInput(session, "route_key_filter", choices = choices, selected = "All")
    })

    survival_data <- reactive({
      req(term())
      get_survival_curves(term(), topic())
    }) |>
      bindCache(term(), topic())

    output$funnel <- plotly::renderPlotly({
      df <- funnel_data()
      if (!nrow(df)) return(plotly::plotly_empty())
      plotly::plot_ly(df, x = ~from, y = ~entered, type = "bar", name = "Entered") |>
        plotly::add_trace(y = ~advanced, name = "Advanced") |>
        plotly::layout(barmode = "group", xaxis = list(title = "Stage"), yaxis = list(title = "Count"))
    })

    output$routes <- DT::renderDT({
      df <- route_data()
      if (!nrow(df)) {
        return(DT::datatable(tibble::tibble(message = "No route archetypes available.")))
      }
      DT::datatable(
        df,
        filter = "top",
        selection = "single",
        extensions = "Buttons",
        options = list(pageLength = 10, dom = "Bfrtip", buttons = c("copy", "csv"))
      )
    })

    observeEvent(input$routes_rows_selected, {
      df <- route_data()
      idx <- input$routes_rows_selected
      if (length(idx) == 1 && idx <= nrow(df)) {
        selected_route(df$route_key[idx])
        updateSelectizeInput(session, "route_key_filter", selected = df$route_key[idx])
      }
    })

    observeEvent(input$route_key_filter, {
      value <- input$route_key_filter
      if (is.null(value) || value == "All") {
        selected_route(NULL)
      } else {
        selected_route(value)
      }
    })

    output$survival <- plotly::renderPlotly({
      df <- survival_data()
      if (!nrow(df)) return(plotly::plotly_empty())
      plotly::plot_ly(df, x = ~t, y = ~survival, color = ~topic, type = "scatter", mode = "lines") |>
        plotly::layout(xaxis = list(title = "Days"), yaxis = list(title = "Survival probability"), legend = list(orientation = "h"))
    })

    bills_data <- reactive({
      req(term())
      get_bills_table(term(), topic(), input$outcome_filter, selected_route())
    }) |>
      bindCache(term(), topic(), input$outcome_filter, selected_route())

    output$bills_table <- DT::renderDT({
      df <- bills_data()
      if (!nrow(df)) {
        return(DT::datatable(tibble::tibble(message = "No bills found for this view.")))
      }
      if (!"bill_link" %in% names(df) && "bill_ID" %in% names(df)) {
        df$bill_link <- sprintf('<a href="https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=%s" target="_blank">view</a>', df$bill_ID)
      }
      DT::datatable(
        df,
        filter = "top",
        selection = "single",
        escape = FALSE,
        extensions = "Buttons",
        options = list(pageLength = 20, dom = "Bfrtip", buttons = c("copy", "csv"))
      )
    })

    bills_proxy <- DT::dataTableProxy("bills_table")
    observeEvent(search(), {
      DT::updateSearch(bills_proxy, keywords = list(global = search()))
    })

    observeEvent(input$bills_table_rows_selected, {
      df <- bills_data()
      idx <- input$bills_table_rows_selected
      if (length(idx) == 1 && idx <= nrow(df)) {
        selected_bill(df$bill_ID[idx])
      }
    })

    output$why_panel <- renderUI({
      bill_id <- selected_bill()
      if (is.null(bill_id)) return(NULL)
      model <- get_per_bill_model(bill_id)
      if (!nrow(model)) {
        return(bslib::card(bslib::card_body(sprintf("No model explanation available for %s", bill_id))))
      }
      info <- model[1, ]
      bottlenecks <- if ("committee_bottlenecks" %in% names(info)) jsonlite::fromJSON(info$committee_bottlenecks) else NULL
      pivotal <- if ("pivotal_actors" %in% names(info)) jsonlite::fromJSON(info$pivotal_actors) else NULL
      bslib::card(
        bslib::card_header(sprintf("Why this bill moves: %s", bill_id)),
        bslib::card_body(
          if (!is.null(info$p_pass_total)) htmltools::tags$p(sprintf("Estimated pass probability: %.1f%%", info$p_pass_total * 100)),
          if (!is.null(bottlenecks)) htmltools::tags$div(
            htmltools::tags$h6("Committee bottlenecks"),
            htmltools::tags$ul(purrr::map(bottlenecks, htmltools::tags$li))
          ),
          if (!is.null(pivotal)) htmltools::tags$div(
            htmltools::tags$h6("Pivotal actors"),
            htmltools::tags$ul(purrr::map(pivotal, htmltools::tags$li))
          )
        )
      )
    })
  })
}

available_terms <- tryCatch(get_available_terms(), error = function(e) character())
if (!length(available_terms)) {
  available_terms <- "2023-2024"
}
DEFAULT_TERM <- available_terms[[1]]
available_topics <- tryCatch(sort(unique(get_topic_summary(0)$topic)), error = function(e) character())
available_topics <- available_topics[!is.na(available_topics) & nzchar(available_topics)]

default_page <- "Overview"

default_state <- function(session, default_term = DEFAULT_TERM) {
  list(
    page = get_query_value(session, "page", default_page),
    term = get_query_value(session, "term", default_term),
    topic = get_query_value(session, "topic", TOPIC_ALL_LABEL),
    search = get_query_value(session, "search", "")
  )
}

enableBookmarking("url")

server <- function(input, output, session) {
  state <- default_state(session, DEFAULT_TERM)

  header_state <- mod_header_server(
    "header",
    terms = available_terms,
    topics = available_topics,
    default_term = state$term,
    default_topic = state$topic,
    default_search = state$search,
    on_state_change = function() {
      update_query(session, list(
        page = input$main_nav,
        term = header_state$term(),
        topic = header_state$topic(),
        search = if (nzchar(header_state$search())) header_state$search() else NULL
      ))
    }
  )

  observeEvent(TRUE, {
    if (!is.null(state$page)) {
      bslib::nav_select(session, "main_nav", state$page)
    }
    header_state$update_term(state$term)
    header_state$update_topic(state$topic)
    header_state$update_search(state$search)
  }, once = TRUE)

  observeEvent(header_state$search(), {
    update_query(session, list(
      page = input$main_nav,
      term = header_state$term(),
      topic = header_state$topic(),
      search = if (nzchar(header_state$search())) header_state$search() else NULL
    ), mode = "replace")
  })

  observeEvent(input$main_nav, {
    update_query(session, list(
      page = input$main_nav,
      term = header_state$term(),
      topic = header_state$topic(),
      search = if (nzchar(header_state$search())) header_state$search() else NULL
    ), mode = "replace")
  })

  mod_overview_server("overview", header_state$term)
  mod_map_server("map", header_state$term, header_state$topic)
  mod_topics_server("topics", header_state$term, header_state$topic)
  mod_people_server("people", header_state$term, header_state$topic, header_state$search)
  mod_donors_server("donors", header_state$term, header_state$topic, header_state$search)
  mod_bills_server("bills", header_state$term, header_state$topic, header_state$search)
}
