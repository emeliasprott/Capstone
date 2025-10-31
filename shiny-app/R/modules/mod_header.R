mod_header_ui <- function(id, terms, topics) {
    ns <- NS(id)
    term_choices <- setNames(terms, terms)
    topic_choices <- c(TOPIC_ALL_LABEL, sort(unique(topics)))
    tagList(
        span(class = "app-title", APP_TITLE),
        bslib::nav_spacer(),
        div(
            class = "header-controls d-flex align-items-center gap-2",
            shiny::selectInput(
                inputId = ns("term"),
                label = shiny::span(class = "visually-hidden", "Select term"),
                choices = term_choices,
                selected = term_choices[1],
                width = "140px"
            ),
            shiny::selectizeInput(
                inputId = ns("topic"),
                label = shiny::span(class = "visually-hidden", "Filter by topic"),
                choices = topic_choices,
                selected = TOPIC_ALL_LABEL,
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

mod_header_server <- function(id, default_term, on_state_change) {
    moduleServer(id, function(input, output, session) {
        term <- shiny::reactiveVal(default_term)
        topic <- shiny::reactiveVal(TOPIC_ALL_LABEL)
        search <- shiny::reactiveVal("")

        observeEvent(input$term,
            {
                term(input$term)
                on_state_change()
            },
            ignoreNULL = FALSE
        )

        observeEvent(input$topic,
            {
                topic(input$topic)
                on_state_change()
            },
            ignoreNULL = FALSE
        )

        observeEvent(input$search,
            {
                search(input$search)
            },
            ignoreNULL = FALSE
        )

        observeEvent(input$copy_link, {
            session$sendCustomMessage("copy-url", list(url = session$clientData$url_href))
        })

        list(
            term = shiny::reactive(term()),
            topic = shiny::reactive(topic()),
            search = shiny::reactive(search()),
            update_term = function(value) {
                if (!is.null(value) && !identical(value, term())) {
                    shiny::updateSelectInput(session, "term", selected = value)
                    term(value)
                }
            },
            update_topic = function(value) {
                if (!is.null(value) && !identical(value, topic())) {
                    shiny::updateSelectizeInput(session, "topic", selected = value)
                    topic(value)
                }
            },
            update_search = function(value) {
                if (!is.null(value) && !identical(value, search())) {
                    shiny::updateTextInput(session, "search", value = value)
                    search(value)
                }
            }
        )
    })
}
