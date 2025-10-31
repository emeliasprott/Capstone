mod_people_ui <- function(id) {
  ns <- NS(id)
  bslib::layout_sidebar(
    sidebar = bslib::card(
      bslib::card_header("Filters"),
      bslib::card_body(
        shiny::radioButtons(ns("people_tab"), "View", choices = c("Legislators", "Committees"), inline = TRUE),
        shiny::checkboxGroupInput(ns("party_filter"), "Party", choices = c("D", "R"), selected = c("D", "R"), inline = TRUE),
        shiny::checkboxGroupInput(ns("chamber_filter"), "Chamber", choices = c("Assembly", "Senate"), selected = c("Assembly", "Senate"), inline = TRUE),
        shiny::helpText("Refine the leaderboard using party, chamber, and the global topic filter above.")
      )
    ),
    bslib::layout_columns(
      bslib::card(
        bslib::card_header("Who stands out"),
        bslib::card_body(
          htmltools::tags$ul(id = ns("summary_points"), class = "mb-3"),
          p("Selections reflect the current term and topic context.")
        )
      ),
      bslib::card(
        bslib::card_header("Leaders"),
        bslib::card_body(DT::DTOutput(ns("leader_table")))
      ),
      bslib::card(
        bslib::card_header("Detail"),
        bslib::card_body(
          plotly::plotlyOutput(ns("trend_chart"), height = 260),
          DT::DTOutput(ns("drill_bills"))
        )
      )
    )
  )
}

mod_people_server <- function(id, term, topic, search) {
  moduleServer(id, function(input, output, session) {
    selected_entity <- shiny::reactiveVal(NULL)

    people_data <- shiny::reactive({
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
      shiny::bindCache(term(), topic(), input$people_tab, input$party_filter, input$chamber_filter)

    summary_points <- shiny::reactive({
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
