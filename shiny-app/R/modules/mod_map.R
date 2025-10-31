mod_map_ui <- function(id) {
  ns <- NS(id)
  bslib::layout_sidebar(
    fillable = TRUE,
    sidebar = bslib::card(
      bslib::card_body(
        shiny::selectInput(
          ns("metric"),
          "Funding metric",
          choices = names(MAP_METRIC_LABELS),
          selected = names(MAP_METRIC_LABELS)[1]
        ),
        shiny::selectizeInput(
          ns("entity_search"),
          "Highlight entity",
          choices = NULL,
          options = list(placeholder = "Search donor or lobby firm"),
          multiple = FALSE
        )
      )
    ),
    bslib::card(
      class = "map-card",
      bslib::card_header("Money by county"),
      bslib::card_body(
        p("County shading shows the selected funding metric for the current term. Hover for rank and percentile; click a county to explore trends."),
        leaflet::leafletOutput(ns("county_map"), height = 520)
      )
    ),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(
        bslib::card_header("County trend"),
        bslib::card_body(plotly::plotlyOutput(ns("county_trends"), height = 320))
      ),
      bslib::card(
        bslib::card_header("Top donors & lobby"),
        bslib::card_body(DT::DTOutput(ns("top_actors")))
      )
    )
  )
}

mod_map_server <- function(id, term, topic) {
  moduleServer(id, function(input, output, session) {
    shapes <- shiny::reactive(get_county_shapes())

    observe({
      req(term())
      entities <- get_geo_spread_entities(term())
      shiny::updateSelectizeInput(session, "entity_search", choices = sort(unique(entities$entity_name)), server = TRUE)
    })

    selected_county <- shiny::reactiveVal(NULL)

    metric_key <- shiny::reactive({
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

    entity_overlay <- shiny::reactive({
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

    map_data <- shiny::reactive({
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
      shiny::bindCache(term(), metric_key(), input$entity_search)

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

    county_timeseries <- shiny::reactive({
      req(selected_county())
      get_county_financials(selected_county())
    }) |>
      shiny::bindCache(selected_county())

    output$county_trends <- plotly::renderPlotly({
      df <- county_timeseries()
      validate(need(nrow(df), "Select a county from the map."))
      plotly::plot_ly(df, x = ~term, y = ~total_donations, type = "scatter", mode = "lines+markers", name = "Donations") |>
        plotly::add_trace(y = ~total_lobbying, name = "Lobbying") |>
        plotly::add_trace(y = ~total_received, name = "Received") |>
        plotly::layout(xaxis = list(title = "Term"), yaxis = list(title = "USD"), legend = list(orientation = "h"))
    })

    top_actors_data <- shiny::reactive({
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
      shiny::bindCache(term(), selected_county())

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
