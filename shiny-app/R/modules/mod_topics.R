mod_topics_ui <- function(id) {
  ns <- NS(id)
  tagList(
    div(class = "mb-3", "Compare issue areas across polarization, party unity, and funding. Click a card for details."),
    bslib::layout_columns(
      col_widths = c(4, 4, 4),
      bslib::card(
        bslib::card_header("Sort topics"),
        bslib::card_body(
          shiny::selectInput(
            ns("sort_by"),
            NULL,
            choices = c("Polarization", "Party-line share", "Controversy", "Momentum"),
            selected = "Polarization"
          )
        )
      ),
      bslib::card(
        bslib::card_header("Minimum roll calls"),
        bslib::card_body(
          shiny::sliderInput(ns("min_rollcalls"), NULL, min = 0, max = 100, value = 10, step = 5)
        )
      ),
      bslib::card(
        bslib::card_header("Topic filter"),
        bslib::card_body(
          shiny::textOutput(ns("active_topic"))
        )
      )
    ),
    shiny::uiOutput(ns("topic_cards")),
    shiny::uiOutput(ns("topic_detail"))
  )
}

mod_topics_server <- function(id, term, topic_filter) {
  moduleServer(id, function(input, output, session) {
    selected_topic <- shiny::reactiveVal(NULL)

    observeEvent(topic_filter(), {
      if (!is.null(topic_filter()) && topic_filter() != TOPIC_ALL_LABEL) {
        selected_topic(topic_filter())
      }
    })

    output$active_topic <- shiny::renderText({
      if (is.null(topic_filter()) || topic_filter() == TOPIC_ALL_LABEL) {
        "All topics"
      } else {
        sprintf("Focused on: %s", topic_filter())
      }
    })

    topics_data <- shiny::reactive({
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
      shiny::bindCache(term(), topic_filter(), input$min_rollcalls, input$sort_by)

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
        shiny::observeEvent(input[[paste0("select_", topic_id)]], {
          selected_topic(topic_name)
        }, ignoreNULL = TRUE)
      })
    })

    output$topic_cards <- shiny::renderUI({
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
        minority <- safe_pull(stats, "minority_cross_rate", NA_real_)
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

    topic_detail_data <- shiny::reactive({
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

    output$topic_detail <- shiny::renderUI({
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
