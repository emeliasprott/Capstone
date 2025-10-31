mod_overview_ui <- function(id) {
  ns <- NS(id)
  bslib::layout_columns(
    col_widths = c(3, 3, 3, 3),
    bslib::card(
      class = "metric-card",
      bslib::card_header("Bills in play"),
      bslib::card_body(
        shiny::uiOutput(ns("bills_value"), class = "metric-value"),
        p("Number of active and archived bills captured in this legislative term."),
        class = "d-flex flex-column gap-2"
      )
    ),
    bslib::card(
      class = "metric-card",
      bslib::card_header("Committees"),
      bslib::card_body(
        shiny::uiOutput(ns("committees_value")),
        p("Committee bodies with measurable gatekeeping or workload activity."),
        class = "d-flex flex-column gap-2"
      )
    ),
    bslib::card(
      class = "metric-card",
      bslib::card_header("Legislators"),
      bslib::card_body(
        shiny::uiOutput(ns("legislators_value")),
        p("Individual lawmakers tracked for influence, voting behavior, and activity."),
        class = "d-flex flex-column gap-2"
      )
    ),
    bslib::card(
      class = "metric-card",
      bslib::card_header("Money movers"),
      bslib::card_body(
        shiny::uiOutput(ns("money_value")),
        p("Distinct donors and lobby firms contributing funds or effort."),
        class = "d-flex flex-column gap-2"
      )
    ),
    bslib::card(
      colspan = 6,
      bslib::card_header("What's in this dashboard"),
      bslib::card_body(
        p("Track legislation, money flows, and actors shaping outcomes. Use the navigation tabs above to explore geography, topics, people, donors, and bill progress."),
        class = "d-flex"
      )
    ),
    bslib::card(
      colspan = 6,
      bslib::card_header("How to read the visuals"),
      bslib::card_body(
        p("Hover for details, use the global term/topic filters, and click rows or cards for deeper context. Tables are searchable and downloadable via the built-in buttons."),
        class = "d-flex"
      )
    )
  )
}

mod_overview_server <- function(id, term) {
  moduleServer(id, function(input, output, session) {
    counts <- shiny::reactive({
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
      shiny::bindCache(term())

    output$bills_value <- shiny::renderUI({
      value <- counts()$n_bills[1]
      span(class = "metric-value", scales::comma(value))
    })

    output$committees_value <- shiny::renderUI({
      value <- counts()$n_committees[1]
      span(class = "metric-value", scales::comma(value))
    })

    output$legislators_value <- shiny::renderUI({
      value <- counts()$n_legislators[1]
      span(class = "metric-value", scales::comma(value))
    })

    output$money_value <- shiny::renderUI({
      donors <- counts()$n_donors[1]
      lobby <- counts()$n_lobby_firms[1]
      span(class = "metric-value", sprintf("%s donors / %s lobby", scales::comma(donors), scales::comma(lobby)))
    })
  })
}
