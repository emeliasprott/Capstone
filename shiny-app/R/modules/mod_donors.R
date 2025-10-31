mod_donors_ui <- function(id) {
  ns <- NS(id)
  tagList(
    div(class = "mb-3", "Explore donors and lobby firms by stance, geographic spread, and total influence."),
    bslib::card(
      bslib::card_body(
        shiny::radioButtons(ns("entity_type"), "Entity type", choices = c("Donor", "Lobby firm"), inline = TRUE),
        shiny::selectizeInput(ns("entity_search"), "Focus on entity", choices = NULL, options = list(placeholder = "Start typing a name"), multiple = FALSE)
      )
    ),
    bslib::card(
      bslib::card_header("Stance vs. geographic spread"),
      bslib::card_body(
        p("Horizontal axis shows how concentrated an entity is geographically (left = concentrated). Vertical axis shows stance score. Circle size scales with totals; color reflects lean."),
        plotly::plotlyOutput(ns("entity_scatter"), height = 420)
      )
    ),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(
        bslib::card_header("Entity list"),
        bslib::card_body(DT::DTOutput(ns("entity_table")))
      ),
      bslib::card(
        bslib::card_header("Entity detail"),
        bslib::card_body(
          leaflet::leafletOutput(ns("entity_map"), height = 320),
          plotly::plotlyOutput(ns("entity_topics"), height = 220)
        )
      )
    )
  )
}

mod_donors_server <- function(id, term, topic, search) {
  moduleServer(id, function(input, output, session) {
    selected_entity <- shiny::reactiveVal(NULL)

    entity_data <- shiny::reactive({
      req(term())
      stance <- get_entity_stance(term())
      if (!nrow(stance)) return(tibble::tibble())
      spread <- get_geo_spread_entities(term())
      df <- dplyr::left_join(stance, spread, by = c("term", "entity_type", "entity_name", "total"))
      df
    }) |>
      shiny::bindCache(term())

    observe({
      df <- entity_data()
      if (!nrow(df)) return(NULL)
      shiny::updateSelectizeInput(session, "entity_search", choices = sort(unique(df$entity_name)), server = TRUE)
    })

    filtered_entities <- shiny::reactive({
      df <- entity_data()
      if (!nrow(df)) return(df)
      df <- df |> dplyr::filter(.data$entity_type == input$entity_type)
      if (!is.null(topic()) && topic() != TOPIC_ALL_LABEL && "topic" %in% names(df)) {
        df <- df |> dplyr::filter(.data$topic == topic())
      }
      df
    }) |>
      shiny::bindCache(term(), input$entity_type, topic())

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
