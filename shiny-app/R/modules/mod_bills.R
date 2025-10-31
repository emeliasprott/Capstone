mod_bills_ui <- function(id) {
  ns <- NS(id)
  tagList(
    bslib::card(
      bslib::card_body(
        shiny::selectInput(ns("outcome_filter"), "Outcome", choices = c("All", "Passed", "Failed"), selected = "All", width = "200px"),
        shiny::selectizeInput(ns("route_key_filter"), "Route archetype", choices = NULL, multiple = FALSE, options = list(placeholder = "All routes"), width = "300px")
      )
    ),
    bslib::layout_columns(
      col_widths = c(7, 5),
      bslib::card(
        bslib::card_header("Bill pipeline"),
        bslib::card_body(plotly::plotlyOutput(ns("funnel"), height = 360))
      ),
      bslib::card(
        bslib::card_header("How to read this"),
        bslib::card_body(p("Each stage shows how many bills entered and advanced. Use the filters to focus on a topic, route, or outcome."))
      )
    ),
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(
        bslib::card_header("Route archetypes"),
        bslib::card_body(DT::DTOutput(ns("routes")))
      ),
      bslib::card(
        bslib::card_header("Survival curve"),
        bslib::card_body(plotly::plotlyOutput(ns("survival"), height = 320))
      )
    ),
    bslib::card(
      bslib::card_header("Bills in motion"),
      bslib::card_body(DT::DTOutput(ns("bills_table")))
    ),
    shiny::uiOutput(ns("why_panel"))
  )
}

mod_bills_server <- function(id, term, topic, search) {
  moduleServer(id, function(input, output, session) {
    selected_route <- shiny::reactiveVal(NULL)
    selected_bill <- shiny::reactiveVal(NULL)

    funnel_data <- shiny::reactive({
      req(term())
      get_pipeline_funnel(term(), topic())
    }) |>
      shiny::bindCache(term(), topic())

    route_data <- shiny::reactive({
      req(term())
      get_route_archetypes(term(), topic())
    }) |>
      shiny::bindCache(term(), topic())

    observe({
      routes <- route_data()
      choices <- c("All", sort(unique(routes$route_key)))
      shiny::updateSelectizeInput(session, "route_key_filter", choices = choices, selected = "All")
    })

    survival_data <- shiny::reactive({
      req(term())
      get_survival_curves(term(), topic())
    }) |>
      shiny::bindCache(term(), topic())

    output$funnel <- plotly::renderPlotly({
      df <- funnel_data()
      if (!nrow(df)) return(plotly::plotly_empty())
      stage_levels <- unique(c(df$from, tail(df$to, 1)))
      df$entered_label <- paste(df$from, "entered")
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
        shiny::updateSelectizeInput(session, "route_key_filter", selected = df$route_key[idx])
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

    bills_data <- shiny::reactive({
      req(term())
      get_bills_table(term(), topic(), input$outcome_filter, selected_route())
    }) |>
      shiny::bindCache(term(), topic(), input$outcome_filter, selected_route())

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

    output$why_panel <- shiny::renderUI({
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
