library(shiny)
library(bslib)
library(DT)
library(plotly)
library(leaflet)
library(purrr)

source("R/globals.R")
source("R/theme.R")
source("R/utils.R")
source("R/load_data.R")
source("R/router.R")

module_files <- list.files("R/modules", full.names = TRUE)
purrr::walk(module_files, source)

TERMS <- tryCatch(get_available_terms(), error = function(e) character())
if (!length(TERMS)) {
    TERMS <- "2023-2024"
}
DEFAULT_TERM <- TERMS[[1]]
TOPICS <- tryCatch(sort(unique(get_topic_summary(0)$topic)), error = function(e) character())


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

ui <- function(request) {
    page <- bslib::page_navbar(
        title = mod_header_ui("header", TERMS, TOPICS),
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

default_page <- "Overview"

default_state <- function(session) {
    list(
        page = get_query_value(session, "page", default_page),
        term = get_query_value(session, "term", DEFAULT_TERM),
        topic = get_query_value(session, "topic", TOPIC_ALL_LABEL),
        search = get_query_value(session, "search", "")
    )
}

server <- function(input, output, session) {
    state <- default_state(session)

    header_state <- mod_header_server("header", state$term, function() {
        update_query(session, list(
            page = input$main_nav,
            term = header_state$term(),
            topic = header_state$topic(),
            search = if (nzchar(header_state$search())) header_state$search() else NULL
        ))
    })

    observeEvent(TRUE,
        {
            if (!is.null(state$page)) {
                bslib::nav_select(session, "main_nav", state$page)
            }
            header_state$update_term(state$term)
            header_state$update_topic(state$topic)
            header_state$update_search(state$search)
        },
        once = TRUE
    )

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

enableBookmarking("url")

shinyApp(ui, server)
