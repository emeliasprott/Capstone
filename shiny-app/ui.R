library(shiny)
library(bslib)
library(DT)
library(plotly)
library(leaflet)

APP_TITLE <- "California Legislative Insights"
TOPIC_ALL_LABEL <- "All topics"
MAP_METRIC_LABELS <- c(
  Donations = "total_donations",
  Lobbying = "total_lobbying",
  Received = "total_received"
)

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

sanitize_id <- function(x) {
  stringr::str_replace_all(tolower(x), "[^a-z0-9]+", "-")
}

app_theme <- function() {
  bslib::bs_theme(
    version = 5,
    preset = "bootstrap",
    primary = "#2B4C7E",
    secondary = "#0F766E",
    success = "#0F766E",
    light = "#f8f9fa",
    dark = "#1f2933"
  )
}

mod_header_ui <- function(id) {
  ns <- NS(id)
  tagList(
    span(class = "app-title", APP_TITLE),
    bslib::nav_spacer(),
    div(
      class = "header-controls d-flex align-items-center gap-2",
      shiny::selectInput(
        inputId = ns("term"),
        label = shiny::span(class = "visually-hidden", "Select term"),
        choices = NULL,
        width = "160px"
      ),
      shiny::selectizeInput(
        inputId = ns("topic"),
        label = shiny::span(class = "visually-hidden", "Filter by topic"),
        choices = NULL,
        options = list(placeholder = "All topics", plugins = list("remove_button")),
        width = "220px"
      ),
      shiny::textInput(
        inputId = ns("search"),
        label = shiny::span(class = "visually-hidden", "Search"),
        placeholder = "Search tables",
        width = "200px"
      ),
      shiny::actionButton(ns("copy_link"), label = NULL, icon = shiny::icon("link"), title = "Copy link to this view"),
      bslib::toggle_dark_mode(id = ns("dark_toggle"), class = "ms-2")
    )
  )
}

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

ui <- function(request) {
  page <- bslib::page_navbar(
    title = mod_header_ui("header"),
    id = "main_nav",
    theme = app_theme(),
    window_title = APP_TITLE,
    collapsible = FALSE,
    bslib::nav_panel("Overview", mod_overview_ui("overview")),
    bslib::nav_panel("Money Map", mod_map_ui("map")),
    bslib::nav_panel("Topics", mod_topics_ui("topics")),
    bslib::nav_panel("People", mod_people_ui("people")),
    bslib::nav_panel("Donors", mod_donors_ui("donors")),
    bslib::nav_panel("Bills", mod_bills_ui("bills")),
    footer = NULL
  )
  tagList(
    page,
    tags$head(tags$link(rel = "stylesheet", href = "css/styles.css")),
    tags$script(HTML(
      "Shiny.addCustomMessageHandler('copy-url', function(message) { navigator.clipboard.writeText(message.url); });"
    ))
  )
}
