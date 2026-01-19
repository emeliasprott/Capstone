library(shiny)
library(shinyWidgets)
library(bslib)
library(DT)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggtext)
library(stringr)
library(lubridate)
library(arrow)
library(sf)
library(leaflet)
library(plotly)
library(purrr)
library(tibble)
library(scales)
library(glue)
library(readr)
library(htmltools)

options(shiny.maxRequestSize = 2000 * 1024^2)

`%||%` <- function(a, b) if (is.null(a)) b else a

parquet_dir <- "../backend/data/outs"
lazy_cache <- new.env(parent = emptyenv())

lazy_load <- function(key, loader) {
    if (!exists(key, envir = lazy_cache, inherits = FALSE)) {
        assign(key, loader(), envir = lazy_cache)
    }
    get(key, envir = lazy_cache, inherits = FALSE)
}

rdp <- function(fname) {
    fpath <- file.path(parquet_dir, fname)
    lazy_load(fpath, function() {
        if (file.exists(fpath)) read_parquet(fpath) else NULL
    })
}

get_counties <- function() {
    lazy_load("ca_counties", function() {
        path <- file.path("../backend/data/ca_counties.geojson")
        if (!file.exists(path)) {
            return(NULL)
        }
        read_sf(path) |>
            st_transform(4326) |>
            mutate(
                county_id = as.integer(county_id),
                county_name = as.character(NAMELSAD)
            ) |>
            select(county_id, county_name, geometry)
    })
}

make_leginfo_link <- function(bill_id_raw) {
    ifelse(
        is.na(bill_id_raw), NA_character_,
        sprintf(
            "<a href='https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=%s' target='_blank'>%s</a>",
            bill_id_raw, bill_id_raw
        )
    )
}

theme_project <- function() {
    theme_minimal() +
        theme(
            plot.title = element_textbox_simple(
                family = "Lato",
                hjust = 0,
                size = 16,
                face = "bold",
                margin = margin(t = 5, b = 6)
            ),
            plot.subtitle = element_textbox_simple(
                family = "Pavanam",
                hjust = 0,
                size = 13,
                color = "#6c757d",
                margin = margin(b = 8, t = 6)
            ),
            legend.title = element_text(family = "Pavanam", size = 11),
            legend.text = element_text(family = "Pavanam", size = 10),
            axis.text.y = element_text(family = "Pavanam", size = 10, color = "#495057"),
            axis.title.y = element_text(family = "Pavanam", size = 11, color = "#343a40"),
            axis.text.x = element_text(family = "Pavanam", size = 10, color = "#495057"),
            axis.title.x = element_text(
                family = "Pavanam", size = 11, color = "#343a40",
                margin = margin(t = 8)
            )
        )
}

theme_app <- bs_theme(
    version = 5,
    bg = "#fbfaf8",
    fg = "#202326",
    primary = "#385f73",
    secondary = "#a9825a",
    "navbar-bg" = "#ffffff",
    "navbar-color" = "#202326",
    "border-color" = "#e4dfd4",
    "card-bg" = "#ffffff",
    "card-border-color" = "#ece7dc",
    "input-border-color" = "#d3cabb",
    "input-border-radius" = "0.5rem",
    "btn-border-radius" = "999px"
)

# ---- PAGE UIs (no sidebar, just stacking cards) ------------------------------

landing_ui <- function() {
    div(
        class = "page-landing",
        div(
            class = "landing-hero section-card-wide",
            fluidRow(
                column(
                    7,
                    h1("See how power moves in Sacramento"),
                    div(
                        class = "landing-text",
                        "This tool weaves together bill content, roll-call votes, campaign money, and lobbying records ",
                        "into a single model of influence. It's built to answer simple questions that usually require ",
                        "hours of digging: Who funds my region? Which topics attract the most money? Whose votes move with that money?"
                    )
                ),
                column(
                    5,
                    div(class = "hero-image")
                )
            )
        ),
        br(),
        div(
            class = "landing-cards-row",
            fluidRow(
                column(
                    4,
                    div(
                        class = "landing-card",
                        h4("Money Map"),
                        p("Zoom into any California county to see modeled flows of campaign and lobbying money, and who's funding your region."),
                        actionLink("go_money", "Go to Money Map", class = "btn-link-card")
                    )
                ),
                column(
                    4,
                    div(
                        class = "landing-card",
                        h4("Bills"),
                        p("Search across sessions and topics to quickly find bills, see modeled signals, and click through to official text."),
                        actionLink("go_bills", "Explore Bills", class = "btn-link-card")
                    )
                ),
                column(
                    4,
                    div(
                        class = "landing-card",
                        h4("Legislators"),
                        p("Compare legislators by fundraising, bill outcomes, and modeled influence at the term or overall level."),
                        actionLink("go_legs", "View Legislators", class = "btn-link-card")
                    )
                )
            ),
            fluidRow(
                column(
                    4,
                    div(
                        class = "landing-card",
                        h4("Donors & Lobbying"),
                        p("See which donors and lobbying firms are most active, how broadly they operate, and which topics they concentrate on."),
                        actionLink("go_donors", "Inspect Donors & Lobbying", class = "btn-link-card")
                    )
                ),
                column(
                    4,
                    div(
                        class = "landing-card",
                        h4("Topics"),
                        p("Drill into modeled issue clusters to see funding intensity, polarization, and the actors driving each topic."),
                        actionLink("go_topics", "Dive into Topics", class = "btn-link-card")
                    )
                )
            )
        )
    )
}

money_ui <- function() {
    div(
        class = "page-money",
        card(
            class = "section-card section-card-wide",
            fluidRow(
                column(
                    8,
                    plotOutput("money_vote_line", height = 280)
                ),
                column(
                    4,
                    div(
                        class = "explain",
                        "Each line compares yes-vote rates for legislators in the top and bottom funding quartiles by term. ",
                        "Use this as context before zooming into specific counties on the map."
                    )
                )
            )
        ),
        card(
            class = "section-card section-card-wide",
            h4(class = "section-title", "Explore funding geographically"),
            fluidRow(
                column(
                    4,
                    pickerInput(
                        "metric_select", "Map metric",
                        choices = c(
                            "Total received" = "total",
                            "Donations" = "donations",
                            "Lobbying" = "lobbying"
                        ),
                        selected = "total",
                        multiple = FALSE,
                        options = pickerOptions(size = 6)
                    )
                ),
                column(
                    8,
                    div(
                        class = "muted",
                        "Each county’s funding is modeled by allocating district-level money using district–county population overlaps. ",
                        "Hover to see amounts by chamber; click a county to populate the table of top funders and topics."
                    )
                )
            )
        ),
        fluidRow(
            column(
                6,
                card(
                    class = "section-card",
                    leafletOutput("money_map", height = 520)
                )
            ),
            column(
                6,
                card(
                    class = "section-card",
                    h4(class = "section-title", "Who funds my region?"),
                    uiOutput("county_selection_label"),
                    DTOutput("county_top_table")
                )
            )
        )
    )
}

bills_ui <- function() {
    div(
        class = "page-bills",
        card(
            class = "section-card section-card-wide",
            h4(class = "section-title", "Bill explorer"),
            div(
                class = "muted",
                "Search across bills (one row per bill) and link directly to official text. ",
                "Use bill IDs for precise lookups, or keywords to search across subjects and descriptions."
            ),
            fluidRow(
                class = "g-2",
                column(
                    5,
                    textInput(
                        "bill_query",
                        label = NULL,
                        placeholder = "Search by bill ID, subject, or keyword...",
                        width = "100%"
                    )
                ),
                column(
                    4,
                    pickerInput(
                        "bill_search_fields",
                        "Search in",
                        multiple = TRUE,
                        selected = c("GeneralSubject"),
                        choices = c(
                            "Bill ID" = "bill_ID",
                            "Raw ID" = "bill_id_raw",
                            "General Subject" = "GeneralSubject"
                        ),
                        options = pickerOptions(actionsBox = TRUE)
                    )
                ),
                column(
                    3,
                    pickerInput(
                        "bill_term_filter",
                        "Term",
                        multiple = TRUE,
                        choices = NULL,
                        options = pickerOptions(
                            actionsBox = TRUE,
                            liveSearch = TRUE,
                            selectedTextFormat = "count > 1"
                        )
                    )
                )
            )
        ),
        card(
            class = "section-card section-card-wide",
            DTOutput("bills_dt")
        )
    )
}

legislators_ui <- function() {
    div(
        class = "page-legislators",
        card(
            class = "section-card section-card-wide",
            h4(class = "section-title", "Legislator outcomes & influence"),
            div(
                class = "muted",
                "View each legislator-term's fundraising, lobbying, and bill outcomes."
            ),
            fluidRow(
                class = "g-2",
                column(
                    4,
                    pickerInput(
                        "leg_single_term",
                        "Term",
                        choices = NULL,
                        options = pickerOptions(liveSearch = TRUE)
                    )
                )
            )
        ),
        card(
            class = "section-card section-card-wide",
            DTOutput("leg_terms_table")
        )
    )
}

donors_ui <- function() {
    div(
        class = "page-donors",
        card(
            class = "section-card section-card-wide",
            h4(class = "section-title", "Top donors and lobby firms"),
            div(
                class = "muted",
                "Use this table to explore major donors and lobbying firms, their total modeled giving, ",
                "how many counties they touch, and their dominant topics."
            ),
            fluidRow(
                class = "g-2",
                column(
                    4,
                    pickerInput(
                        "dl_kind",
                        "Type",
                        multiple = TRUE,
                        selected = c("Donor", "Lobbying"),
                        choices = c("Donor", "Lobbying")
                    )
                )
            )
        ),
        card(
            class = "section-card section-card-wide",
            DTOutput("donor_lobby_table")
        )
    )
}

topics_ui <- function() {
    div(
        class = "page-topics",
        card(
            class = "section-card section-card-wide",
            h4(class = "section-title", "Topic drilldown"),
            div(
                class = "muted",
                "Topics are modeled clusters of bills and actors. Select a topic and one or more terms to see summary statistics ",
                "and the key legislators, donors, and firms driving it."
            ),
            fluidRow(
                class = "toolbar-row g-2",
                column(
                    4,
                    pickerInput(
                        "topic_pick",
                        "Topic",
                        choices = NULL,
                        options = pickerOptions(liveSearch = TRUE)
                    )
                ),
                column(
                    4,
                    pickerInput(
                        "term_multi",
                        "Terms",
                        multiple = TRUE,
                        choices = NULL,
                        options = pickerOptions(
                            actionsBox = TRUE,
                            liveSearch = TRUE,
                            selectedTextFormat = "count > 2"
                        )
                    )
                )
            ),
            uiOutput("topic_stats"),
            uiOutput("topic_top_actors")
        ),
        fluidRow(
            column(
                7,
                card(
                    class = "section-card",
                    h4(class = "section-title", "Funding intensity by topic & term"),
                    plotlyOutput("topic_heat", height = 360)
                )
            ),
            column(
                5,
                card(
                    class = "section-card",
                    div(
                        class = "muted",
                        "Heat map of total modeled funding (donations + lobbying) by topic and term. ",
                        "Darker cells indicate topic–term pairs that attract more money."
                    )
                )
            )
        ),
        fluidRow(
            column(
                7,
                card(
                    class = "section-card",
                    h4(class = "section-title", "Polarization & success landscape"),
                    plotlyOutput("topic_bubbles", height = 360)
                )
            ),
            column(
                5,
                card(
                    class = "section-card",
                    div(
                        class = "muted",
                        "Each bubble is a topic. Vertical axis: average bill success; horizontal axis: vote entropy / partisan alignment; ",
                        "size: number of bills; color: average polarization."
                    )
                )
            )
        )
    )
}

# ---- MAIN UI (header with nav, no tabs) --------------------------------------

ui <- page_fillable(
    theme = theme_app,
    tags$head(
        tags$link(href = "https://fonts.googleapis.com", rel = "preconnect"),
        tags$link(
            href = "https://fonts.gstatic.com",
            rel = "preconnect",
            crossorigin = "crossorigin"
        ),
        tags$link(
            href = "https://fonts.googleapis.com/css2?family=Lato:wght@300;400;600;700&family=Lusitana:wght@400;700&family=Pavanam&display=swap",
            rel = "stylesheet"
        ),
        tags$link(rel = "stylesheet", type = "text/css", href = "styles.css"),
        tags$title("California Legislative Insights")
    ),
    div(
        class = "app-shell",
        div(
            class = "app-header-bar",
            div(class = "app-title", "California Legislative Insights"),
            div(class = "app-header-nav", uiOutput("header_nav"))
        ),
        uiOutput("main_page")
    )
)

# ---- SERVER -------------------------------------------------------------------

server <- function(input, output, session) {
    current_page <- reactiveVal("landing")

    # header nav (reactive active state)
    output$header_nav <- renderUI({
        pg <- current_page()
        mk <- function(id, label) {
            cls <- "header-nav-link"
            if (id == pg) cls <- paste(cls, "active")
            actionLink(paste0("nav_", id), label = label, class = cls)
        }

        tagList(
            mk("landing", "Welcome"),
            mk("money", "Money Map"),
            mk("bills", "Bills"),
            mk("legislators", "Legislators"),
            mk("donors", "Donors & Lobbying"),
            mk("topics", "Topics")
        )
    })

    # router
    output$main_page <- renderUI({
        switch(current_page(),
            "landing" = landing_ui(),
            "money" = money_ui(),
            "bills" = bills_ui(),
            "legislators" = legislators_ui(),
            "donors" = donors_ui(),
            "topics" = topics_ui(),
            landing_ui()
        )
    })

    # nav from landing cards
    observeEvent(input$go_money, current_page("money"))
    observeEvent(input$go_bills, current_page("bills"))
    observeEvent(input$go_legs, current_page("legislators"))
    observeEvent(input$go_donors, current_page("donors"))
    observeEvent(input$go_topics, current_page("topics"))

    # nav from header
    observeEvent(input$nav_landing, current_page("landing"))
    observeEvent(input$nav_money, current_page("money"))
    observeEvent(input$nav_bills, current_page("bills"))
    observeEvent(input$nav_legislators, current_page("legislators"))
    observeEvent(input$nav_donors, current_page("donors"))
    observeEvent(input$nav_topics, current_page("topics"))

    # ---- Money vs vote line plot -----------------------------------------------
    output$money_vote_line <- renderPlot({
        df <- rdp("money_vote_alignment.parquet")
        if (is.null(df)) {
            return(NULL)
        }

        df <- df |>
            mutate(term_year = as.integer(sub("-.*$", "", as.character(term)))) |>
            arrange(term_year)

        ggplot(df, aes(x = term_year, group = 1)) +
            geom_line(aes(y = yes_rate_top, color = "Top funded"), linewidth = 1) +
            geom_line(aes(y = yes_rate_bottom, color = "Bottom funded"), linewidth = 1) +
            labs(y = "Yes vote rate", title = "Legislators Above the 75th Percentile in Funding Have Voting Trends Almost Opposite that of the 25th Percentile", subtitle = "Measured by proportion of yes votes relative to the total") +
            scale_y_continuous(labels = percent_format(accuracy = 1)) +
            scale_color_manual(
                values = c("Top funded" = "#385f73", "Bottom funded" = "#d4a15f"),
                guide = guide_legend(title = NULL)
            ) +
            theme_project() +
            theme(legend.position = "right", axis.title.x = element_blank())
    })

    # ---- Money map -------------------------------------------------------------
    wide_data <- reactive({
        reg_funds <- rdp("ca_legislator_funding.parquet")
        counties <- get_counties()
        if (is.null(reg_funds) || is.null(counties)) {
            return(NULL)
        }

        reg_funds |>
            mutate(
                county_id = as.integer(county_id),
                house = tolower(house)
            ) |>
            group_by(county_id, house) |>
            summarise(
                donations = sum(total_donations, na.rm = TRUE),
                lobbying = sum(total_lobbying, na.rm = TRUE),
                total = sum(total_received, na.rm = TRUE),
                .groups = "drop"
            ) |>
            pivot_wider(
                id_cols = county_id,
                names_from = house,
                values_from = c(donations, lobbying, total),
                names_glue = "{house}_{.value}"
            ) |>
            mutate(
                don_sum = rowSums(across(ends_with("_donations")), na.rm = TRUE),
                lob_sum = rowSums(across(ends_with("_lobbying")), na.rm = TRUE),
                tot_sum = rowSums(across(ends_with("_total")), na.rm = TRUE)
            ) |>
            left_join(counties, by = "county_id") |>
            st_as_sf()
    })

    output$money_map <- renderLeaflet({
        df <- wide_data()
        if (is.null(df) || nrow(df) == 0) {
            return(NULL)
        }

        metric <- switch(input$metric_select,
            "donations" = "don_sum",
            "lobbying" = "lob_sum",
            "total" = "tot_sum",
            "tot_sum"
        )

        vals <- df[[metric]]
        pal <- colorQuantile("Blues", domain = vals, n = 6, na.color = "#f5f5f5")
        df$fill_value <- vals

        assembly_col <- paste0("assembly_", input$metric_select)
        senate_col <- paste0("senate_", input$metric_select)

        safe_col <- function(col) if (col %in% names(df)) df[[col]] else rep(0, nrow(df))

        labels <- sprintf(
            "<strong>%s</strong><br/>Assembly %s: %s<br/>Senate&nbsp;&nbsp;%s: %s<br/><span class='label-strong'>Combined total:</span> %s",
            df$county_name %||% "",
            tools::toTitleCase(input$metric_select),
            dollar(safe_col(assembly_col)),
            tools::toTitleCase(input$metric_select),
            dollar(safe_col(senate_col)),
            dollar(df[[metric]] %||% 0)
        ) |>
            lapply(HTML)

        leaflet(df, options = leafletOptions(minZoom = 5, maxZoom = 12)) |>
            addProviderTiles(providers$CartoDB.Positron) |>
            addPolygons(
                fillColor = ~ pal(fill_value),
                weight = 0.4,
                color = "#c2b8aa",
                fillOpacity = 0.9,
                label = labels,
                labelOptions = labelOptions(
                    style = list(
                        "font-size" = "11px",
                        "font-family" = "Pavanam"
                    )
                ),
                layerId = ~county_id,
                highlightOptions = highlightOptions(
                    weight = 1.2,
                    color = "#385f73",
                    fillOpacity = 0.98,
                    bringToFront = TRUE
                )
            ) |>
            addLegend(
                pal = pal,
                values = vals,
                title = paste(tools::toTitleCase(input$metric_select), "($)"),
                position = "bottomright",
                labFormat = labelFormat(prefix = "$")
            ) |>
            fitBounds(lng1 = -124, lat1 = 32, lng2 = -119.0, lat2 = 42.1)
    })

    selected_county <- reactiveVal(NULL)

    observeEvent(input$money_map_shape_click,
        {
            click <- input$money_map_shape_click
            if (!is.null(click$id)) selected_county(as.integer(click$id))
        },
        ignoreInit = TRUE
    )

    county_top_funders <- reactive({
        df <- rdp("county_top_funders.parquet")
        if (is.null(df)) {
            return(NULL)
        }
        labels <- sapply(df$top_topics_labels, function(x) paste(x, collapse = ", "))
        df$top_topics <- labels
        df |>
            mutate(
                county_id = as.integer(county_id),
                rank_in_county = as.integer(rank_in_county)
            )
    })

    county_selected_df <- reactive({
        cid <- selected_county()
        ctf <- county_top_funders()
        if (is.null(cid) || is.null(ctf)) {
            return(tibble())
        }
        ctf |>
            filter(county_id == cid) |>
            arrange(rank_in_county)
    })

    observeEvent(wide_data(),
        {
            df <- wide_data()
            if (is.null(df)) {
                return()
            }
            df2 <- st_drop_geometry(df)
            if (!"tot_sum" %in% names(df2) || nrow(df2) == 0) {
                return()
            }
            top_county <- df2$county_id[which.max(df2$tot_sum)]
            if (!is.null(top_county) && is.null(selected_county())) {
                selected_county(as.integer(top_county))
            }
        },
        once = TRUE,
        ignoreInit = FALSE
    )

    output$county_selection_label <- renderUI({
        df <- county_selected_df()
        if (nrow(df) == 0) {
            return(div(class = "muted", "Select a county on the map to see top funders and topics."))
        }
        nm <- df$county_name[1] %||% "Selected county"
        div(
            class = "muted",
            HTML(paste0("<b>", nm, "</b>: Top funders and topics based on modeled county-level flows."))
        )
    })

    output$county_top_table <- renderDT({
        df <- county_selected_df()
        if (nrow(df) == 0) {
            return(
                datatable(
                    tibble(Note = "Select a county with available data to see top funders."),
                    rownames = FALSE,
                    options = list(dom = "t", paging = FALSE),
                    escape = TRUE,
                    class = "display nowrap compact stripe"
                )
            )
        }

        df <- df |>
            mutate(total_amount = dollar(round(total_amount, 2))) |>
            select(
                Rank = rank_in_county,
                Funder = funder,
                Type = kind,
                `Total Amount` = total_amount,
                `Top Topics` = top_topics
            )

        datatable(
            df,
            rownames = FALSE,
            options = list(
                dom = "t",
                pageLength = 10,
                scrollX = TRUE,
                ordering = FALSE,
                autoWidth = TRUE,
                columnDefs = list(list(class = "funders-col", targets = 1))
            ),
            escape = FALSE,
            class = "display nowrap compact stripe"
        )
    })

    # ---- Bills ------------------------------------------------------------------
    observe({
        df <- rdp("bills_table.parquet")
        if (is.null(df) || !"term" %in% names(df)) {
            return()
        }
        updatePickerInput(
            session, "bill_term_filter",
            choices = sort(unique(as.character(df$term)))
        )
    })

    bills_filtered <- reactive({
        df <- rdp("bills_table.parquet")
        if (is.null(df)) {
            return(tibble())
        }

        df <- df |>
            mutate(
                bill_link = make_leginfo_link(bill_ID),
                term = as.character(term)
            )

        q <- trimws(input$bill_query %||% "")
        fields <- input$bill_search_fields %||% character()
        terms_sel <- input$bill_term_filter %||% character()

        if (!nzchar(q) && length(terms_sel) == 0) {
            return(
                df |>
                    select(
                        bill_link, topic, term, outcome,
                        yes, no,
                        First_action, longevity_days, n_versions,
                        mean_yes_ratio_versions,
                        bill_polarization,
                        bill_controversiality,
                        bill_vote_entropy
                    )
            )
        }

        dat <- df

        if (length(terms_sel) > 0 && "term" %in% names(dat)) {
            dat <- dat |> filter(term %in% terms_sel)
        }

        if (nzchar(q) && length(fields) > 0) {
            q_low <- tolower(q)
            is_billish <- grepl("^[AS][BJ]\\s*\\d+", toupper(q)) || grepl("^\\d{4}", q)

            mask_list <- lapply(fields, function(f) {
                col <- dat[[f]]
                if (is.null(col)) {
                    return(rep(FALSE, nrow(dat)))
                }
                x <- tolower(as.character(col %||% ""))

                if (f %in% c("bill_ID", "bill_id") && is_billish) {
                    startsWith(gsub("\\s", "", x), gsub("\\s", "", q_low))
                } else if (f == "GeneralSubject") {
                    toks <- unlist(strsplit(q_low, "\\s+"))
                    rowSums(sapply(toks, function(tk) grepl(tk, x, fixed = TRUE))) >= length(toks)
                } else {
                    grepl(q_low, x, fixed = TRUE)
                }
            })

            if (length(mask_list) > 0) {
                mask <- Reduce("|", mask_list)
                dat <- dat[mask, , drop = FALSE]
            }
        }

        dat |>
            select(
                `Bill` = bill_link,
                topic,
                term,
                outcome,
                yes,
                no,
                First_action,
                longevity_days,
                n_versions,
                mean_yes_ratio_versions,
                bill_polarization,
                bill_controversiality,
                bill_vote_entropy
            )
    })

    output$bills_dt <- renderDT({
        dat <- bills_filtered()
        datatable(
            dat,
            escape = FALSE,
            rownames = FALSE,
            extensions = c("Scroller", "FixedHeader"),
            options = list(
                dom = "t",
                deferRender = TRUE,
                scrollY = 520,
                scroller = TRUE,
                scrollX = TRUE,
                fixedHeader = TRUE,
                pageLength = 25,
                ordering = TRUE
            ),
            class = "display nowrap stripe compact"
        )
    })

    # ---- Legislators -----------------------------------------------------------
    leg_terms_data <- reactive({
        df <- rdp("leg_terms.parquet")
        if (is.null(df) || !"term" %in% names(df)) {
            return(tibble())
        }
        df |>
            mutate(term = as.character(term))
    })

    observe({
        lt <- leg_terms_data()
        if (nrow(lt) == 0) {
            return()
        }
        updatePickerInput(
            session, "leg_single_term",
            choices = sort(unique(lt$term))
        )
    })

    leg_filtered <- reactive({
        lt <- leg_terms_data()
        if (nrow(lt) == 0) {
            return(tibble())
        }



        if (is.null(input$leg_single_term)) {
            return(tibble())
        }
        lt |> filter(term == input$leg_single_term)
    })

    output$leg_terms_table <- renderDT({
        df <- leg_filtered()
        if (nrow(df) == 0 || !"full_name" %in% names(df)) {
            return(
                datatable(
                    tibble(Note = "Legislator data unavailable or incomplete."),
                    rownames = FALSE,
                    options = list(dom = "t", paging = FALSE),
                    escape = TRUE,
                    class = "display nowrap compact"
                )
            )
        }


        out <- df |>
            rename(Legislator = full_name, Term = term) |>
            group_by(Legislator, Term) |>
            summarize(
                Party = first(Party %||% NA_character_),
                House = first(chamber %||% NA_character_),
                `Total Donations` = dollar(sum(total_donations %||% 0, na.rm = TRUE)),
                `Total Lobbying` = dollar(sum(total_lobbying %||% 0, na.rm = TRUE)),
                `Bills Passed` = percent(sum(outcome %||% 0, na.rm = TRUE)),
                .groups = "drop_last"
            ) |>
            arrange(House, Party, Legislator, Term)

        datatable(
            out,
            rownames = FALSE,
            options = list(
                dom = "t",
                pageLength = 25,
                ordering = TRUE,
                scrollX = TRUE
            ),
            escape = FALSE,
            class = "display nowrap stripe compact"
        )
    })

    # ---- Donors / Lobby firms --------------------------------------------------
    donor_lobby_data <- reactive({
        ctf <- rdp("county_top_funders.parquet")
        if (is.null(ctf)) {
            return(tibble())
        }

        ctf |>
            transmute(
                funder,
                kind,
                county = county_name,
                total_amount,
                top_topics
            ) |>
            group_by(funder, kind) |>
            summarize(
                total_amount = sum(total_amount, na.rm = TRUE),
                n_counties = n_distinct(county),
                top_topics = paste(na.omit(unique(top_topics)), collapse = "; "),
                .groups = "drop"
            )
    })

    donor_lobby_filtered <- reactive({
        df <- donor_lobby_data()
        if (nrow(df) == 0) {
            return(df)
        }
        kinds <- input$dl_kind %||% c("Donor", "Lobbying")
        df |> filter(kind %in% kinds)
    })

    output$donor_lobby_table <- renderDT({
        df <- donor_lobby_filtered()
        if (nrow(df) == 0) {
            return(
                datatable(
                    tibble(Note = "No donor/lobby data available."),
                    rownames = FALSE,
                    options = list(dom = "t", paging = FALSE),
                    escape = TRUE,
                    class = "display nowrap compact"
                )
            )
        }

        datatable(
            df |>
                arrange(desc(total_amount)),
            rownames = FALSE,
            options = list(
                dom = "t",
                pageLength = 25,
                scrollX = TRUE
            ),
            escape = FALSE,
            class = "display nowrap stripe compact"
        )
    })

    # ---- Topics -----------------------------------------------------------------
    observe({
        tfbt <- rdp("topic_funding_by_term.parquet")
        if (is.null(tfbt) || !all(c("label", "term") %in% names(tfbt))) {
            return()
        }
        updatePickerInput(
            session, "topic_pick",
            choices = sort(unique(tfbt$label))
        )
        updatePickerInput(
            session, "term_multi",
            choices = sort(unique(as.character(tfbt$term)))
        )
    })

    output$topic_heat <- renderPlotly({
        tfbt <- rdp("topic_funding_by_term.parquet")
        if (is.null(tfbt) || !all(c("topic", "term", "total_received") %in% names(tfbt))) {
            return(NULL)
        }

        mat <- tfbt |>
            mutate(term = as.character(term)) |>
            group_by(label, term) |>
            summarize(
                total = sum(total_received, na.rm = TRUE),
                .groups = "drop"
            ) |>
            tidyr::complete(label, term, fill = list(total = 0))

        plot_ly(
            mat,
            x = ~term,
            y = ~label,
            z = ~total,
            type = "heatmap",
            colors = "Blues"
        ) |>
            layout(
                margin = list(l = 140, r = 20, t = 20, b = 60),
                xaxis = list(title = "Term"),
                yaxis = list(title = "Topic")
            )
    })

    output$topic_bubbles <- renderPlotly({
        bi <- rdp("bills_table.parquet")
        if (is.null(bi) || !all(c(
            "label",
            "bill_polarization",
            "mean_yes_ratio_versions",
            "bill_vote_entropy"
        ) %in% names(bi))) {
            return(NULL)
        }

        dd <- bi |>
            group_by(label) |>
            summarize(
                n_bills = n(),
                mean_polarization = mean(bill_polarization, na.rm = TRUE),
                success_rate = mean(mean_yes_ratio_versions, na.rm = TRUE),
                partisan_align = mean(bill_vote_entropy, na.rm = TRUE),
                .groups = "drop"
            )

        plot_ly(
            dd,
            x = ~partisan_align,
            y = ~success_rate,
            size = ~n_bills,
            color = ~mean_polarization,
            type = "scatter",
            mode = "markers",
            sizes = c(8, 40),
            marker = list(sizemode = "area")
        ) |>
            layout(
                xaxis = list(title = "Avg vote entropy"),
                yaxis = list(title = "Avg bill success rate"),
                margin = list(l = 80, r = 20, t = 20, b = 60)
            )
    })

    output$topic_stats <- renderUI({
        if (is.null(input$topic_pick) || is.null(input$term_multi)) {
            return(NULL)
        }
        bi <- rdp("bills_table.parquet")
        if (is.null(bi)) {
            return(NULL)
        }

        dd <- bi |>
            filter(
                label == input$topic_pick,
                term %in% paste(
                    as.character(as.integer(input$term_multi)),
                    as.character(as.integer(input$term_multi) + 1),
                    sep = "-"
                )
            )

        if (nrow(dd) == 0) {
            return(div(class = "explain muted", "No bills found for this topic/term selection."))
        }

        rate <- mean(dd$mean_yes_ratio_versions, na.rm = TRUE)
        pol <- mean(dd$bill_polarization, na.rm = TRUE)
        cont <- mean(dd$bill_controversiality, na.rm = TRUE)

        div(
            class = "explain",
            HTML(glue(
                "<b>Topic {input$topic_pick}</b>: pass rate ",
                "<b>{scales::percent(rate %||% 0, accuracy = 0.1)}</b>, ",
                "avg polarization <b>{scales::number(pol %||% 0, accuracy = 0.01)}</b>, ",
                "avg controversiality <b>{scales::number(cont %||% 0, accuracy = 0.01)}</b>."
            ))
        )
    })

    output$topic_top_actors <- renderUI({
        if (is.null(input$topic_pick) || is.null(input$term_multi)) {
            return(NULL)
        }

        tfbl <- rdp("topic_funding_by_leg.parquet")
        dtt <- rdp("donor_topic_by_term.parquet")
        lbt <- rdp("lobby_firm_topic_by_term.parquet")

        pieces <- list()

        if (!is.null(tfbl)) {
            top_legs <- tfbl |>
                mutate(term = as.character(term)) |>
                filter(
                    label == input$topic_pick,
                    term %in% input$term_multi
                ) |>
                group_by(canon) |>
                summarize(total = sum(total, na.rm = TRUE), .groups = "drop") |>
                arrange(desc(total)) |>
                slice_head(n = 5)
            if (nrow(top_legs) > 0) {
                pieces <- c(pieces, list(
                    div(
                        class = "explain",
                        HTML(
                            "<b>Top legislators for this topic:</b> ",
                            paste(top_legs$canon, collapse = ", ")
                        )
                    )
                ))
            }
        }

        if (!is.null(dtt)) {
            top_don <- dtt |>
                filter(
                    label == input$topic_pick,
                    term %in% input$term_multi
                ) |>
                group_by(ExpenderName) |>
                summarize(total = sum(donations_allocated, na.rm = TRUE), .groups = "drop") |>
                arrange(desc(total)) |>
                slice_head(n = 5)
            if (nrow(top_don) > 0) {
                pieces <- c(pieces, list(
                    div(
                        class = "explain",
                        HTML(
                            "<b>Top donors:</b> ",
                            paste(top_don$ExpenderName, collapse = ", ")
                        )
                    )
                ))
            }
        }

        if (!is.null(lbt)) {
            top_lf <- lbt |>
                filter(
                    label == input$topic_pick,
                    term %in% input$term_multi
                ) |>
                group_by(FIRM_NAME) |>
                summarize(total = sum(lobby_allocated, na.rm = TRUE), .groups = "drop") |>
                arrange(desc(total)) |>
                slice_head(n = 5)
            if (nrow(top_lf) > 0) {
                pieces <- c(pieces, list(
                    div(
                        class = "explain",
                        HTML(
                            "<b>Top lobby firms:</b> ",
                            paste(top_lf$FIRM_NAME, collapse = ", ")
                        )
                    )
                ))
            }
        }

        if (length(pieces) == 0) {
            return(NULL)
        }
        tagList(pieces)
    })
}

shinyApp(ui, server)
