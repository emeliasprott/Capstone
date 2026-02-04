library(shiny)
library(bslib)
library(shinyWidgets)
library(DT)
library(arrow)
library(sf)
library(tidyverse)
library(leaflet)
library(scales)
library(stringr)
library(ggtext)
library(ggrepel)
library(plotly)

options(shiny.fullstacktrace = TRUE)
options(shiny.sanitize.errors = FALSE)

`%||%` <- function(a, b) if (!is.null(a)) a else b

# ----------------------------
# Paths + lazy parquet loader
# ----------------------------
parquet_dir <- "data"
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
        path <- "data/ca_counties.geojson"

        read_sf(path) |>
            st_transform(4326)
    })
}


make_leginfo_link <- function(bill_id_raw) {
    ifelse(
        is.na(bill_id_raw) | bill_id_raw == "",
        NA_character_,
        sprintf(
            "<a href='https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=%s' target='_blank'>%s</a>",
            bill_id_raw, bill_id_raw
        )
    )
}

# ----------------------------
# Theme
# ----------------------------
theme_app <- bs_theme(
    version = 5,
    bg = "#f8f9fa",
    fg = "#1a1a1a",
    primary = "#2c5f7d",
    secondary = "#7a5c47",
    success = "#2d7a4f",
    info = "#3b82c5",
    warning = "#d97706",
    danger = "#dc2626",
    "navbar-bg" = "#ffffff",
    "navbar-color" = "#1a1a1a",
    "border-color" = "#e5e7eb",
    "card-bg" = "#ffffff",
    "card-border-color" = "#e5e7eb",
    "card-border-radius" = "12px",
    "input-border-color" = "#d1d5db",
    "input-border-radius" = "8px",
    "btn-border-radius" = "8px",
    base_font = font_google("Inter"),
    heading_font = font_google("Libre Franklin")
)

theme_project <- function(base_size = 13) {
    theme_minimal(base_size = base_size) +
        theme(
            text = element_text(family = "Inter", color = "#1a1a1a"),
            plot.title = element_textbox_simple(
                family = "Inter",
                face = "bold",
                size = 18,
                color = "#1a1a1a",
                margin = margin(b = 8)
            ),
            plot.subtitle = element_textbox_simple(
                family = "Inter",
                size = 13,
                color = "#828282",
                margin = margin(b = 12),
                lineheight = 1.4
            ),
            plot.caption = element_text(
                size = 11,
                color = "#acacac",
                hjust = 0,
                margin = margin(t = 10)
            ),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_line(color = "#f3f4f6", linewidth = 0.5),
            axis.title.x = element_text(
                margin = margin(t = 12),
                size = 12,
                color = "#616161",
                face = "bold"
            ),
            axis.title.y = element_text(
                margin = margin(r = 12),
                size = 12,
                color = "#616161",
                face = "bold"
            ),
            axis.text = element_text(size = 11, color = "#7a7a7a"),
            axis.ticks = element_line(color = "#eaeaea"),
            plot.background = element_rect(fill = "white", color = NA),
            panel.background = element_rect(fill = "white", color = NA),
            legend.position = "bottom",
            legend.title = element_text(face = "bold", size = 11),
            legend.text = element_text(size = 10)
        )
}

# ----------------------------
# UI pieces
# ----------------------------
landing_ui <- function() {
    div(
        class = "page-landing",
        div(
            class = "landing-hero",
            div(
                class = "hero-content",
                div(
                    class = "hero-badge",
                    icon("chart-line"),
                    span("California Legislative Power & Policy Explorer")
                ),
                h1(class = "hero-title", "How Money, Power, and Policy Interact in California"),
                p(
                    class = "hero-subtitle",
                    "An interactive view of California's legislative process combining campaign finance, lobbying activity, voting behavior, and policy outcomes to reveal who influences what, where, and why.",
                ),
                div(
                    class = "hero-stats",
                    div(
                        class = "stat-item",
                        div(class = "stat-value", "58"),
                        div(class = "stat-label", "Counties")
                    ),
                    div(
                        class = "stat-item",
                        div(class = "stat-value", "25+"),
                        div(class = "stat-label", "Years of Data")
                    ),
                    div(
                        class = "stat-item",
                        div(class = "stat-value", "63,100"),
                        div(class = "stat-label", "Bills Analyzed")
                    )
                )
            )
        ),
        div(
            class = "landing-cards",
            div(
                class = "feature-card",
                div(class = "card-icon", icon("map-marked-alt")),
                h3("Regional Analysis"),
                p("Explore funding exposure by county and discover which organizations are shaping policy in each region."),
                actionLink("go_regions", "Explore Regions", class = "card-link", icon = icon("arrow-right"))
            ),
            div(
                class = "feature-card",
                div(class = "card-icon", icon("money-bill-wave")),
                h3("Funding Patterns"),
                p("Understand how campaign contributions flow through the legislature and who the major players are."),
                actionLink("go_funding", "View Funding", class = "card-link", icon = icon("arrow-right"))
            ),
            div(
                class = "feature-card",
                div(class = "card-icon", icon("fire")),
                h3("Policy Topics"),
                p("Track controversy trends and identify where money and political attention concentrate."),
                actionLink("go_topics", "Analyze Topics", class = "card-link", icon = icon("arrow-right"))
            ),
            div(
                class = "feature-card",
                div(class = "card-icon", icon("file-alt")),
                h3("Bill Explorer"),
                p("Search and filter thousands of bills by keyword, author, and legislative session."),
                actionLink("go_bills", "Search Bills", class = "card-link", icon = icon("arrow-right"))
            )
        )
    )
}

regions_ui <- function() {
    div(
        class = "page-regions",
        div(
            class = "page-header",
            h2(class = "page-title", icon("map-marked-alt"), "Regional Analysis"),
            p(
                class = "page-description",
                "County shading represents modeled legislative funding exposure on a percentile scale. ",
                "Select a county to explore the top funders influencing that region."
            ),
            div(
                class = "info-badge small regions", icon("info-circle"),
                "Includes reported campaign contributions and lobbying activity only. Ballot-measure spending is excluded."
            )
        ),
        div(
            class = "content-grid regions-grid",
            div(
                class = "grid-main",
                card(
                    class = "viz-card",
                    div(
                        class = "card-header-custom",
                        h4("Funding Distribution by County"),
                        div(
                            class = "info-badge", icon("info-circle"),
                            "Click any county to view details"
                        )
                    ),
                    leafletOutput("money_map", height = 600)
                )
            ),
            div(
                class = "grid-sidebar regions",
                card(
                    class = "detail-card county-detail-card",
                    div(
                        class = "card-header-custom",
                        h4("Top Funders"),
                        uiOutput("county_selection_label")
                    ),
                    DTOutput("county_top_table")
                )
            )
        )
    )
}

funding_ui <- function() {
    div(
        class = "page-funding",
        div(
            class = "page-header",
            h2(class = "page-title", icon("money-bill-wave"), "Funding Analysis"),
            p(
                class = "page-description",
                "Explore how campaign contributions are distributed across legislators and identify key funding patterns."
            )
        ),
        card(
            class = "viz-card section-card-wide",
            div(
                class = "card-header-custom",
                h4("Funding Distribution Across Legislators")
            ),
            fluidRow(
                column(
                    8,
                    plotOutput("funding_dist", height = 380)
                ),
                column(
                    4,
                    div(
                        class = "insight-box",
                        div(class = "insight-icon", icon("lightbulb")),
                        div(
                            class = "insight-content",
                            h5("What does this mean?"),
                            p(
                                "Most legislators fall into the central, more normal range of funding, but the highest funding is concentrated around a small group."
                            )
                        )
                    )
                )
            )
        ),
        card(
            class = "data-card section-card-wide funders-card",
            div(
                class = "card-header-custom",
                h4("Complete Funding Records")
            ),
            DTOutput("funders_dt")
        )
    )
}

topics_ui <- function() {
    div(
        class = "page-topics",
        div(
            class = "page-header",
            h2(class = "page-title", icon("fire"), "Policy Topics"),
            p(
                class = "page-description",
                "Compare policy areas by controversy, funding intensity, and polarization to see which topics dominate legislative attention."
            )
        ),
        card(
            class = "viz-card section-card-wide",
            div(
                class = "card-header-custom",
                h4("Controversy Trends Over Time")
            ),
            fluidRow(
                column(
                    8,
                    plotOutput("topic_controversy_lines", height = 400)
                ),
                column(
                    4,
                    div(
                        class = "control-panel",
                        h5("Customize View"),
                        p(
                            class = "control-description",
                            "Select up to five policy areas to compare their controversy trajectories. By default, the most volatile topics are shown."
                        ),
                        pickerInput(
                            "topic_highlight",
                            "Select Topics",
                            choices = NULL,
                            multiple = TRUE,
                            options = pickerOptions(
                                liveSearch = TRUE,
                                actionsBox = TRUE,
                                selectedTextFormat = "count > 2",
                                liveSearchPlaceholder = "Search topics..."
                            )
                        )
                    )
                )
            )
        ),
        card(
            class = "viz-card section-card-wide",
            div(
                class = "card-header-custom",
                h4("Topic Polarization")
            ),
            plotlyOutput("topic_polarization", height = 500)
        ),
        div(
            class = "metrics-grid",
            card(
                class = "metric-card",
                div(
                    class = "card-header-custom",
                    h4(icon("chart-line"), "Controversy Shifts"),
                    p(class = "card-subtitle", "Topics with largest changes in controversiality")
                ),
                plotOutput("topic_delta_bar", height = 360)
            ),
            card(
                class = "metric-card",
                div(
                    class = "card-header-custom",
                    h4(icon("dollar-sign"), "Top Funded Topics"),
                    p(class = "card-subtitle", "Recent legislative sessions (2023-2025)")
                ),
                plotOutput("topic_money_bar", height = 360)
            )
        ),
        div(
            class = "metrics-grid",
            card(
                class = "metric-card",
                div(
                    class = "card-header-custom",
                    h4(icon("exclamation-triangle"), "Most Controversial"),
                    p(class = "card-subtitle", "Based on voting patterns and procedural intensity")
                ),
                plotOutput("topic_controversy_bar", height = 360)
            ),
            card(
                class = "metric-card hotspot-card",
                div(
                    class = "card-header-custom",
                    h4(icon("crosshairs"), "Policy Hotspots"),
                    p(class = "card-subtitle", "High controversy + high funding")
                ),
                DTOutput("latest_conflict", height = 360)
            )
        )
    )
}

bills_ui <- function() {
    div(
        class = "page-bills",
        div(
            class = "page-header",
            h2(class = "page-title", icon("file-alt"), "Bill Explorer"),
            p(
                class = "page-description",
                "Search and filter California legislation by keyword, author, bill ID, and legislative session."
            )
        ),
        card(
            class = "search-card section-card-wide",
            div(
                class = "card-header-custom",
                h4("Search Criteria"),
                p(class = "card-subtitle", "All fields are optional—use any combination to find bills")
            ),
            div(
                class = "search-grid",
                div(
                    class = "search-field",
                    div(class = "search-label", icon("search"), "Keyword"),
                    textInput("bill_keyword",
                        label = NULL,
                        placeholder = "e.g., housing, wildfire, climate"
                    )
                ),
                div(
                    class = "search-field",
                    div(class = "search-label", icon("user"), "Author"),
                    textInput("bill_author",
                        label = NULL,
                        placeholder = "e.g., Smith, Garcia"
                    )
                ),
                div(
                    class = "search-field",
                    div(class = "search-label", icon("hashtag"), "Bill ID"),
                    textInput("bill_id",
                        label = NULL,
                        placeholder = "e.g., AB123, SB456"
                    )
                ),
                div(
                    class = "search-field",
                    div(class = "search-label", icon("calendar"), "Legislative Terms"),
                    pickerInput(
                        "bill_terms",
                        label = NULL,
                        choices = NULL,
                        multiple = TRUE,
                        options = pickerOptions(
                            actionsBox = TRUE,
                            liveSearch = TRUE,
                            selectedTextFormat = "count > 2",
                            liveSearchPlaceholder = "Search terms..."
                        )
                    )
                )
            )
        ),
        card(
            class = "results-card section-card-wide bills-card",
            div(
                class = "card-header-custom",
                h4("Search Results"),
                div(
                    class = "results-info",
                    icon("info-circle"),
                    "Controversy percentile shows how contentious a bill was relative to others in its term"
                )
            ),
            DTOutput("bills_dt")
        )
    )
}

# ----------------------------
# Main UI shell (routing)
# ----------------------------
ui <- page_fillable(
    theme = theme_app,
    tags$head(
        tags$link(href = "https://fonts.googleapis.com", rel = "preconnect"),
        tags$link(href = "https://fonts.gstatic.com", rel = "preconnect", crossorigin = "crossorigin"),
        tags$link(
            href = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Libre+Franklin:ital,wght@0,100..900;1,100..900&display=swap",
            rel = "stylesheet"
        ),
        tags$link(
            rel = "stylesheet",
            href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
        ),
        tags$link(rel = "stylesheet", type = "text/css", href = "style.css"),
        tags$title("California Legislative Power & Policy Explorer"),
        tags$meta(name = "viewport", content = "width=device-width, initial-scale=1")
    ),
    div(
        class = "app-shell",
        tags$header(
            class = "app-header",
            div(
                class = "header-container",
                div(
                    class = "app-brand",
                    div(class = "brand-icon", icon("landmark")),
                    div(
                        class = "brand-text",
                        div(class = "brand-title", "California Legislative Policy Explorer"),
                        div(class = "brand-subtitle", "Legislative finance, voting, and policy dynamics - mapped and measured")
                    )
                ),
                tags$nav(class = "app-nav", uiOutput("header_nav"))
            )
        ),
        tags$main(class = "app-main", uiOutput("main_page")),
        tags$footer(
            class = "app-footer",
            div(
                class = "footer-content",
                a(href = "https://github.com/emeliasprott/Capstone", "Learn more about how this works"),
                p("Data updated through the end of 2025"),
                p(class = "footer-meta", "Built with R Shiny")
            )
        )
    )
)

# ----------------------------
# Search helpers
# ----------------------------
tokenize_simple <- function(x) {
    x <- tolower(x %||% "")
    x <- gsub("[^a-z0-9\\s]", " ", x)
    x <- gsub("\\s+", " ", x)
    trimws(x)
}

bm25_build <- function(docs_tokens) {
    n <- length(docs_tokens)
    dfreq <- new.env(parent = emptyenv())
    dl <- lengths(docs_tokens)
    avgdl <- mean(dl)

    for (i in seq_len(n)) {
        u <- unique(docs_tokens[[i]])
        for (t in u) {
            cur <- get0(t, envir = dfreq, ifnotfound = 0)
            assign(t, cur + 1, envir = dfreq)
        }
    }

    terms <- ls(dfreq, all.names = TRUE)
    idf <- setNames(numeric(length(terms)), terms)
    for (t in terms) {
        dft <- get(t, envir = dfreq)
        idf[[t]] <- log((n - dft + 0.5) / (dft + 0.5) + 1)
    }

    list(idf = idf, avgdl = avgdl, dl = dl)
}

bm25_score_one <- function(doc_tokens, query_tokens, idf, avgdl, dl, k1 = 1.2, b = 0.75) {
    if (length(query_tokens) == 0) {
        return(0)
    }
    tf <- table(doc_tokens)
    score <- 0
    for (t in query_tokens) {
        if (!t %in% names(tf)) next
        f <- as.numeric(tf[[t]])
        denom <- f + k1 * (1 - b + b * (dl / avgdl))
        score <- score + (idf[[t]] %||% 0) * (f * (k1 + 1) / denom)
    }
    score
}

listcol_to_str <- function(x, max_n = 6) {
    if (is.null(x)) {
        return("")
    }
    if (!is.list(x)) {
        return(as.character(x %||% ""))
    }
    v <- unlist(x)
    v <- v[!is.na(v)]
    if (length(v) == 0) {
        return("")
    }
    if (length(v) > max_n) v <- c(v[1:max_n], "…")
    paste(v, collapse = ", ")
}

# ----------------------------
# Server
# ----------------------------
server <- function(input, output, session) {
    current_page <- reactiveVal("landing")

    output$header_nav <- renderUI({
        pg <- current_page()
        mk <- function(id, label, icon_name) {
            cls <- "nav-item"
            if (id == pg) cls <- paste(cls, "active")
            actionLink(
                paste0("nav_", id),
                label = tagList(icon(icon_name), span(label)),
                class = cls
            )
        }
        tagList(
            mk("landing", "Home", "home"),
            mk("regions", "Regions", "map-marked-alt"),
            mk("funding", "Funding", "money-bill-wave"),
            mk("topics", "Topics", "fire"),
            mk("bills", "Bills", "file-alt")
        )
    })

    output$main_page <- renderUI({
        switch(current_page(),
            "landing" = landing_ui(),
            "regions" = regions_ui(),
            "funding" = funding_ui(),
            "topics" = topics_ui(),
            "bills" = bills_ui(),
            landing_ui()
        )
    })

    observeEvent(input$go_regions, current_page("regions"))
    observeEvent(input$go_funding, current_page("funding"))
    observeEvent(input$go_topics, current_page("topics"))
    observeEvent(input$go_bills, current_page("bills"))

    observeEvent(input$nav_landing, current_page("landing"))
    observeEvent(input$nav_regions, current_page("regions"))
    observeEvent(input$nav_funding, current_page("funding"))
    observeEvent(input$nav_topics, current_page("topics"))
    observeEvent(input$nav_bills, current_page("bills"))

    # ----------------------------
    # Regions
    # ----------------------------
    county_sf <- reactive({
        counties <- get_counties()
        county_funding <- rdp("county_funding.parquet")

        counties |>
            left_join(
                county_funding,
                by = c("county_id", "NAMELSAD" = "county_name")
            ) |>
            mutate(
                funding_quantile = percent_rank(total_amount)
            )
    })


    selected_county <- reactiveVal(NULL)

    observeEvent(input$money_map_shape_click,
        {
            click <- input$money_map_shape_click
            if (is.null(click$id)) {
                return()
            }
            df <- county_sf()
            nm <- df$NAMELSAD[df$county_id == as.integer(click$id)][1]

            selected_county(nm)
        },
        ignoreInit = TRUE
    )


    output$money_map <- renderLeaflet({
        df <- county_sf()
        pal <- colorNumeric(
            palette = c("#d7eae9", "#bcdcda", "#a1cecc", "#72b5b3", "#4d9391", "#386b69", "#2A504F", "#234342"),
            domain = df$funding_quantile,
            na.color = "#f3f4f6"
        )

        leaflet(df) |>
            addProviderTiles(providers$CartoDB.Positron) |>
            addPolygons(
                fillColor = ~ pal(funding_quantile),
                fillOpacity = 0.8,
                color = "#ffffff",
                weight = 1.5,
                opacity = 1,
                layerId = ~county_id,
                label = ~NAMELSAD,
                highlightOptions = highlightOptions(
                    weight = 3,
                    color = "#4d9391",
                    fillOpacity = 0.9,
                    bringToFront = TRUE
                )
            ) |>
            addLegend(
                pal = pal,
                values = ~funding_quantile,
                title = "Funding Intensity",
                opacity = 1,
                position = "bottomright",
                labFormat = labelFormat(
                    transform = function(x) x * 100,
                    suffix = "%"
                )
            )
    })


    county_top_tbl <- reactive({
        df <- rdp("county_top_funders.parquet")
        if (is.null(df)) {
            return(NULL)
        }

        as_tibble(df) |>
            mutate(
                county_id = as.integer(county_id),
                total_amount = as.numeric(total_amount)
            )
    })

    county_term <- reactive({
        df <- rdp("county_term_funding.parquet")
        if (is.null(df)) {
            return(NULL)
        }

        as_tibble(df)
    })


    county_selected_df <- reactive({
        nm <- selected_county()
        df <- county_top_tbl()

        if (is.null(nm) || is.null(df)) {
            return(tibble())
        }

        df |>
            filter(county_name == nm) |>
            arrange(desc(total_amount))
    })


    output$county_selection_label <- renderUI({
        nm <- selected_county()
        if (is.null(nm)) {
            div(
                class = "selection-prompt",
                icon("hand-pointer"),
                span("Click a county on the map to view details")
            )
        } else {
            div(
                class = "selected-county",
                icon("map-marker-alt"),
                strong(nm)
            )
        }
    })


    output$county_top_table <- renderDT({
        df <- county_selected_df()
        if (nrow(df) == 0) {
            return(datatable(
                tibble(Note = "No county selected"),
                rownames = FALSE,
                options = list(dom = "t", paging = FALSE)
            ))
        }

        datatable(
            df |>
                transmute(
                    Funder = str_to_title(str_to_lower(funder)),
                    `Total Amount` = dollar(total_amount),
                    `Top Supported Topics` = top_supported_topics,
                    `Top Opposed Topics` = top_opposed_topics,
                    `Concentration in Region` = scales::percent(regional_concentration, scale = 1)
                ),
            rownames = FALSE,
            options = list(
                dom = "t",
                scrollX = TRUE,
                ordering = FALSE
            ),
            class = "display compact stripe"
        )
    })



    # ----------------------------
    # Funding
    # ----------------------------
    leg_funding <- reactive({
        rdp("legislator_funding.parquet")
    })

    output$funding_dist <- renderPlot({
        df <- leg_funding()
        if (is.null(df) || nrow(df) == 0) {
            return(NULL)
        }

        q50 <- quantile(df$total_funding + 1, 0.5, na.rm = TRUE)
        q90 <- quantile(df$total_funding + 1, 0.9, na.rm = TRUE)

        ggplot(df, aes(x = total_funding + 1)) +
            geom_histogram(
                bins = 35,
                fill = "#4d9391",
                color = "white",
                linewidth = 0.3,
                alpha = 0.9
            ) +
            scale_x_log10(
                labels = dollar_format(),
                breaks = c(10, 100, 1e3, 1e4, 1e5, 1e6)
            ) +
            geom_vline(
                xintercept = c(q50, q90),
                linetype = "dashed",
                color = "#72b5b3",
                linewidth = 1,
                alpha = 0.8
            ) +
            annotate(
                geom = "label",
                x = q50, y = Inf,
                vjust = 1.3, hjust = -0.05,
                label = "Median",
                color = "#445050",
                fill = "white",
                size = 3.8,
                fontface = "bold",
                label.padding = unit(0.3, "lines")
            ) +
            annotate(
                geom = "label",
                x = q90, y = Inf,
                vjust = 1.3, hjust = -0.05,
                label = "90th percentile",
                color = "#445050",
                fill = "white",
                size = 3.8,
                fontface = "bold",
                label.padding = unit(0.3, "lines")
            ) +
            labs(
                x = "Total Funding Received (log scale)",
                y = "Number of Legislators"
            ) +
            theme_project(13)
    })

    output$funders_dt <- renderDT({
        df <- rdp("funding.parquet")
        if (is.null(df)) {
            return(datatable(
                rownames = FALSE,
                options = list(dom = "t", paging = FALSE),
                class = "display nowrap compact stripe"
            ))
        }

        out <- df %>%
            mutate(
                amount = dollar(round(amount, 2)),
                House = str_to_title(house),
                Type = str_to_title(kind)
            ) %>%
            select(
                Firm,
                Amount = amount,
                Term = term,
                Party = party,
                House,
                Name
            )

        datatable(
            out,
            rownames = FALSE,
            escape = FALSE,
            extensions = "Buttons",
            options = list(
                pageLength = 15,
                scrollX = FALSE,
                autoWidth = TRUE,
                dom = "<'dt-top dt-top-split'Bf>t<'dt-bottom'lp>",
                buttons = list(
                    list(
                        extend = "csv",
                        text = "<i class='fa-solid fa-download'></i> Download",
                        exportOptions = list(modifier = list(page = "all"))
                    )
                )
            ),
            class = "display compact stripe"
        )
    })

    # ----------------------------
    # Topics
    # ----------------------------
    topic_term <- reactive({
        df <- rdp("topic_term_summary.parquet")
        if (is.null(df)) {
            return(NULL)
        }
        df |>
            mutate(
                year = as.integer(str_split_i(as.character(term), "-", 1))
            )
    })

    topic_term_indexed <- reactive({
        df <- topic_term()
        if (is.null(df) || nrow(df) == 0) {
            return(NULL)
        }
        df |>
            group_by(topic) |>
            mutate(
                controversy_index = avg_controversy / mean(avg_controversy, na.rm = TRUE)
            ) |>
            ungroup()
    })

    topic_defaults <- reactive({
        df <- topic_term_indexed()
        if (is.null(df) || nrow(df) == 0) {
            return(list(all = character(), default = character(), terms = character()))
        }

        topic_slopes <- df |>
            group_by(topic) |>
            summarise(
                movement = sum(abs(diff(controversy_index)), na.rm = TRUE),
                .groups = "drop"
            ) |>
            arrange(desc(movement))

        list(
            all = sort(unique(df$topic)),
            default = topic_slopes |> slice_head(n = 5) |> pull(topic),
            terms = sort(unique(df$term))
        )
    })


    observeEvent(
        list(topic_defaults(), current_page()),
        {
            if (current_page() != "topics") {
                return()
            }

            x <- topic_defaults()
            if (length(x$all) == 0) {
                return()
            }

            updatePickerInput(
                session,
                "topic_highlight",
                choices = x$all,
                selected = x$default
            )

            updatePickerInput(
                session,
                "topic_single",
                choices = x$all,
                selected = x$default[1]
            )

            updatePickerInput(
                session,
                "topic_terms_single",
                choices = x$terms,
                selected = tail(x$terms, 6)
            )
        },
        ignoreInit = TRUE
    )

    observeEvent(input$topic_highlight,
        {
            sel <- input$topic_highlight
            if (length(sel) > 5) {
                updatePickerInput(
                    session,
                    "topic_highlight",
                    selected = sel[1:5]
                )
            }
        },
        ignoreInit = TRUE
    )



    output$topic_controversy_lines <- renderPlot({
        df <- topic_term_indexed()
        if (is.null(df) || nrow(df) == 0) {
            return(NULL)
        }

        sel <- input$topic_highlight

        highlight_topics <- if (is.null(sel) || length(sel) == 0) {
            topic_defaults()$default
        } else {
            sel
        }

        ggplot(df, aes(x = term, y = controversy_index, group = topic)) +
            geom_line(
                data = df |> filter(!topic %in% highlight_topics),
                color = "#eaeaeb",
                linewidth = 0.5,
                alpha = 0.6
            ) +
            geom_line(
                data = df |> filter(topic %in% highlight_topics),
                aes(color = topic),
                linewidth = 1.4
            ) +
            geom_hline(
                yintercept = 1,
                linetype = "dashed",
                color = "#b0b0b0",
                linewidth = 0.7
            ) +
            scale_color_manual(values = c("#72B5B3", "#89A894", "#EDD892", "#A37B73", "#3F292B")) +
            labs(
                title = "Policy Controversy Over Time",
                subtitle = "Indexed to each topic's historical average (baseline = 1.0)",
                x = NULL,
                y = "Controversy Index",
                color = "Selected Topics"
            ) +
            theme_project(13) +
            theme(
                legend.position = "bottom",
                legend.text = element_text(size = 10),
                axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
                plot.margin = margin(10, 20, 10, 10)
            ) +
            guides(color = guide_legend(nrow = 2))
    })
    output$topic_polarization <- renderPlotly({
        df <- topic_term()

        topic_stability <- df %>%
            group_by(topic) %>%
            summarise(
                avg_polarization = mean(topic_polarization, na.rm = TRUE),
                sd_polarization = sd(topic_polarization, na.rm = TRUE),
                n_terms = n(),
                .groups = "drop"
            ) %>%
            filter(n_terms >= 2)

        x_ref <- median(topic_stability$avg_polarization, na.rm = TRUE)
        y_ref <- median(topic_stability$sd_polarization, na.rm = TRUE)
        x_min <- min(topic_stability$avg_polarization, na.rm = TRUE)
        x_max <- max(topic_stability$avg_polarization, na.rm = TRUE)
        y_min <- min(topic_stability$sd_polarization, na.rm = TRUE)
        y_max <- max(topic_stability$sd_polarization, na.rm = TRUE)

        x_pad <- 0.1 * (x_max - x_min)
        y_pad <- 0.1 * (y_max - y_min)

        topic_stability$intensity <- sqrt(
            (topic_stability$avg_polarization / max(topic_stability$avg_polarization, na.rm = TRUE))^2 +
                (topic_stability$sd_polarization / max(topic_stability$sd_polarization, na.rm = TRUE))^2
        )

        plot_ly(
            data = topic_stability,
            x = ~avg_polarization,
            y = ~sd_polarization,
            type = "scatter",
            mode = "markers",
            marker = list(
                size = 10,
                color = ~intensity,
                colorscale = list(
                    c(0, "#D7EAE9"),
                    c(0.3, "#a1cecc"),
                    c(0.6, "#72b5b3"),
                    c(1, "#315E5C")
                ),
                showscale = TRUE,
                colorbar = list(
                    title = list(
                        text = "Conflict<br>Intensity",
                        font = list(
                            family = "SF Pro Display, -apple-system, system-ui, sans-serif",
                            size = 12,
                            color = "#445050"
                        )
                    ),
                    thickness = 15,
                    len = 0.7,
                    x = 1.02,
                    tickfont = list(
                        size = 10,
                        color = "#445050"
                    )
                ),
                line = list(
                    width = 1.5,
                    color = "rgba(255, 255, 255, 0.8)"
                ),
                opacity = 0.85
            ),
            customdata = I(Map(
                function(t, n) list(t, n),
                topic_stability$topic,
                topic_stability$n_terms
            )),
            hovertemplate = paste(
                "<b style='font-size: 13px; color: #1A237E;'>%{customdata[0]}</b><br>",
                "<span style='color: #4B5858;'><b>Average Polarization:</b> %{x:.3f}</span><br>",
                "<span style='color: #4B5858;'><b>Volatility:</b> %{y:.3f}</span><br>",
                "<span style='color: #828a8a;'>Terms Observed: %{customdata[1]}</span>",
                "<extra></extra>"
            )
        ) %>%
            layout(
                title = list(
                    text = "<sub style='font-size: 14px; color: #828a8a;'>Structural vs. Episodic Political Conflict</sub>",
                    x = 0.5,
                    xanchor = "center",
                    font = list(
                        family = "SF Pro Display, -apple-system, system-ui, sans-serif",
                        size = 18
                    ),
                    pad = list(t = 10, b = 10)
                ),
                xaxis = list(
                    title = list(
                        text = "<b>Average Polarization →</b><br><sub style='font-size: 11px;'>Structural Conflict Intensity</sub>",
                        font = list(
                            family = "SF Pro Display, -apple-system, system-ui, sans-serif",
                            size = 13,
                            color = "#37474F"
                        )
                    ),
                    range = c(x_min - x_pad, x_max + x_pad),
                    showgrid = TRUE,
                    gridcolor = "rgba(176, 190, 197, 0.15)",
                    gridwidth = 1,
                    zeroline = FALSE,
                    tickfont = list(
                        size = 11,
                        color = "#828a8a",
                        family = "SF Mono, Consolas, monospace"
                    ),
                    showline = TRUE,
                    linecolor = "rgba(120, 144, 156, 0.3)",
                    linewidth = 1.5
                ),
                yaxis = list(
                    title = list(
                        text = "<b>Volatility Across Terms →</b>",
                        font = list(
                            family = "SF Pro Display, -apple-system, system-ui, sans-serif",
                            size = 13,
                            color = "#4B5858"
                        )
                    ),
                    range = c(y_min - (2.5 * y_pad), y_max + y_pad),
                    showgrid = TRUE,
                    gridcolor = "rgba(176, 190, 197, 0.15)",
                    gridwidth = 1,
                    zeroline = FALSE,
                    tickfont = list(
                        size = 11,
                        color = "#828a8a",
                        family = "SF Mono, Consolas, monospace"
                    ),
                    showline = TRUE,
                    linecolor = "rgba(120, 144, 156, 0.3)",
                    linewidth = 1.5
                ),
                shapes = list(
                    list(
                        type = "line",
                        x0 = x_ref, x1 = x_ref,
                        y0 = 0, y1 = max(topic_stability$sd_polarization, na.rm = TRUE) * 1.05,
                        line = list(
                            color = "rgba(63, 81, 181, 0.4)",
                            dash = "dash",
                            width = 2
                        ),
                        layer = "below"
                    ),
                    list(
                        type = "line",
                        x0 = 0, x1 = 1,
                        y0 = y_ref, y1 = y_ref,
                        line = list(
                            color = "rgba(63, 81, 181, 0.4)",
                            dash = "dash",
                            width = 2
                        ),
                        layer = "below"
                    )
                ),
                annotations = list(
                    list(
                        x = 0.25,
                        y = 0.15,
                        text = "<b>High Structural &<br>High Episodic</b><br><span style='font-size: 10px; color: #a1cecc;'>Persistently divisive<br>& highly volatile</span>",
                        showarrow = FALSE,
                        font = list(
                            size = 11,
                            color = "#4B5858",
                            family = "SF Pro Display, -apple-system, system-ui, sans-serif"
                        ),
                        bgcolor = "rgba(255, 255, 255, 0.85)",
                        bordercolor = "rgba(114, 181, 179, 0.3)",
                        borderwidth = 1,
                        borderpad = 6,
                        xanchor = "center"
                    ),
                    list(
                        x = 0.05,
                        y = 0.15,
                        text = "<b>Low Structural &<br>High Episodic</b><br><span style='font-size: 10px; color: #a1cecc;'>Event-driven<br>polarization</span>",
                        showarrow = FALSE,
                        font = list(
                            size = 11,
                            color = "#4B5858",
                            family = "SF Pro Display, -apple-system, system-ui, sans-serif"
                        ),
                        bgcolor = "rgba(255, 255, 255, 0.85)",
                        bordercolor = "rgba(114, 181, 179, 0.3)",
                        borderwidth = 1,
                        borderpad = 6,
                        xanchor = "center"
                    ),
                    list(
                        x = 0.25,
                        y = 0.03,
                        text = "<b>High Structural &<br>Low Episodic</b><br><span style='font-size: 10px; color: #a1cecc;'>Consistently<br>polarized</span>",
                        showarrow = FALSE,
                        font = list(
                            size = 11,
                            color = "#4B5858",
                            family = "SF Pro Display, -apple-system, system-ui, sans-serif"
                        ),
                        bgcolor = "rgba(255, 255, 255, 0.85)",
                        bordercolor = "rgba(114, 181, 179, 0.3)",
                        borderwidth = 1,
                        borderpad = 6,
                        xanchor = "center"
                    ),
                    list(
                        x = 0.05,
                        y = 0.03,
                        text = "<b>Low Structural &<br>Low Episodic</b><br><span style='font-size: 10px; color: #a1cecc;'>Consensus<br>topics</span>",
                        showarrow = FALSE,
                        font = list(
                            size = 11,
                            color = "#4B5858",
                            family = "SF Pro Display, -apple-system, system-ui, sans-serif"
                        ),
                        bgcolor = "rgba(255, 255, 255, 0.85)",
                        bordercolor = "rgba(114, 181, 179, 0.3)",
                        borderwidth = 1,
                        borderpad = 6,
                        xanchor = "center"
                    ),
                    list(
                        x = x_ref,
                        y = -0.02,
                        text = sprintf("Median: %.3f", x_ref),
                        showarrow = FALSE,
                        font = list(
                            size = 9,
                            color = "#828a8a",
                            family = "SF Mono, Consolas, monospace"
                        ),
                        xanchor = "center",
                        yshift = -20
                    ),
                    list(
                        x = 0.01,
                        y = y_ref,
                        text = sprintf("Median: %.3f", y_ref),
                        showarrow = FALSE,
                        font = list(
                            size = 9,
                            color = "#828a8a",
                            family = "SF Mono, Consolas, monospace"
                        ),
                        yanchor = "bottom",
                        xshift = 20
                    )
                ),
                margin = list(l = 90, r = 120, t = 100, b = 80),
                paper_bgcolor = "#FAFBFC",
                plot_bgcolor = "#FFFFFF",
                hovermode = "closest",
                hoverlabel = list(
                    bgcolor = "rgba(255, 255, 255, 0.95)",
                    bordercolor = "#386b69",
                    font = list(
                        family = "SF Pro Display, -apple-system, system-ui, sans-serif",
                        size = 12,
                        color = "#445050"
                    ),
                    align = "left"
                ),
                plot_bgcolor = "#FFFFFF"
            ) %>%
            config(
                displayModeBar = FALSE,
                displaylogo = FALSE
            )
    })

    output$topic_delta_bar <- renderPlot({
        df <- topic_term()
        if (is.null(df) || nrow(df) == 0) {
            return(NULL)
        }

        d <- df |>
            group_by(topic) |>
            summarise(delta_controversy = last(avg_controversy) - first(avg_controversy), .groups = "drop") |>
            slice_max(abs(delta_controversy), n = 12) |>
            mutate(
                topic = fct_reorder(topic, delta_controversy),
                direction = ifelse(delta_controversy > 0, "Increased", "Decreased")
            )

        ggplot(d, aes(y = topic, x = delta_controversy, fill = direction)) +
            geom_vline(xintercept = 0, linewidth = 0.8, color = "#d1d5db") +
            geom_col(alpha = 0.9, width = 0.7) +
            scale_fill_manual(values = c("Increased" = "#715056", "Decreased" = "#a1cecc")) +
            scale_x_continuous(labels = number_format(accuracy = 0.01)) +
            labs(
                x = "Change in Controversy Score",
                y = NULL,
                fill = "Trend"
            ) +
            theme_project(12) +
            theme(
                panel.grid.major.y = element_blank(),
                legend.position = "top",
                axis.text.y = element_text(size = 10)
            )
    })


    output$topic_money_bar <- renderPlot({
        df <- topic_term()
        if (is.null(df) || nrow(df) == 0) {
            return(NULL)
        }

        latest <- df |>
            filter(year %in% c(2023, 2025)) |>
            group_by(topic) |>
            summarise(topic_amount = sum(topic_amount, na.rm = TRUE), .groups = "drop") |>
            arrange(desc(topic_amount)) |>
            slice_head(n = 12) |>
            mutate(
                topic = fct_reorder(topic, topic_amount)
            )

        ggplot(latest, aes(x = topic_amount, y = topic)) +
            geom_col(fill = "#72b5b3", alpha = 0.9, width = 0.7) +
            geom_text(
                aes(label = dollar(topic_amount, accuracy = 1)),
                hjust = -0.1,
                size = 3.2,
                color = "#445050",
                fontface = "bold"
            ) +
            scale_x_continuous(
                labels = dollar_format(),
                expand = expansion(mult = c(0.02, 0.25))
            ) +
            labs(x = "Total Funding", y = NULL) +
            theme_project(12) +
            theme(
                panel.grid.major.y = element_blank(),
                axis.text.y = element_text(size = 10)
            )
    })


    output$topic_controversy_bar <- renderPlot({
        df <- topic_term()
        if (is.null(df) || nrow(df) == 0) {
            return(NULL)
        }

        latest <- df |>
            filter(year %in% c(2023, 2025)) |>
            group_by(topic) |>
            summarise(avg_controversy = mean(avg_controversy, na.rm = TRUE), .groups = "drop") |>
            arrange(desc(avg_controversy)) |>
            slice_head(n = 15) |>
            mutate(topic = fct_reorder(topic, avg_controversy))

        ggplot(latest, aes(x = avg_controversy, y = topic)) +
            geom_segment(
                aes(x = 0, xend = avg_controversy, yend = topic),
                linewidth = 1.2,
                alpha = 0.6,
                color = "#A37B73"
            ) +
            geom_point(size = 4, color = "#A37B73", alpha = 0.9) +
            labs(x = "Average Controversy Score", y = NULL) +
            theme_project(12) +
            theme(
                panel.grid.major.y = element_blank(),
                axis.text.y = element_text(size = 10)
            )
    })


    output$topic_funding_ts <- renderPlot({
        df <- topic_term()
        if (is.null(df) || nrow(df) == 0) {
            return(NULL)
        }

        topic_name <- input$topic_single
        if (is.null(topic_name) || topic_name == "") {
            return(NULL)
        }

        d <- df |>
            filter(topic == topic_name, !is.na(topic_amount)) |>
            arrange(term)

        if (!is.null(input$topic_terms_single) && length(input$topic_terms_single) > 0) {
            d <- d |>
                filter(term %in% input$topic_terms_single) |>
                arrange(term)
        }

        ggplot(d, aes(x = term, y = topic_amount, group = 1)) +
            geom_line(linewidth = 1.3, color = "#386b69") +
            geom_point(size = 3, color = "#386b69") +
            scale_y_continuous(
                labels = dollar_format(),
                expand = expansion(mult = c(0.05, 0.1))
            ) +
            labs(
                title = paste("Funding Trends:", topic_name),
                subtitle = "Total funding allocated based on legislator topic alignment",
                x = NULL,
                y = "Total Funding"
            ) +
            theme_project(12) +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))
    })

    output$latest_conflict <- renderDT({
        df <- rdp("latest_high_conflict.parquet") %>%
            select(`Topic`, `Total Funding`, `Relative Controversy`, `Pass Rate`, `Funding Percentile`) %>%
            mutate(`Total Funding` = scales::dollar(round(`Total Funding`, 2)), `Relative Controversy` = scales::percent(`Relative Controversy`, scale = 100), `Pass Rate` = scales::percent(`Pass Rate`, scale = 1), `Funding Percentile` = scales::percent(`Funding Percentile`)) %>%
            arrange(desc(`Total Funding`)) %>%
            head(7)
        datatable(
            df,
            rownames = FALSE,
            options = list(dom = "t", scrollX = TRUE, ordering = FALSE),
            class = "display compact stripe"
        )
    })

    # ----------------------------
    # Bills
    # ----------------------------
    bills <- reactive({
        df <- rdp("bill_stats.parquet")
        if (is.null(df)) {
            return(NULL)
        }

        tok_col <- "author_tokens"

        df |>
            mutate(
                term = as.character(term),
                Subject = as.character(Subject),
                Name = as.character(Name),
                bill_link = make_leginfo_link(bill_ID),
                subject_clean = tokenize_simple(Subject),
                subject_tokens = strsplit(subject_clean, " "),
                .tok = if (!is.null(tok_col)) .data[[tok_col]] else list()
            ) |>
            distinct()
    })

    bm25_obj <- reactiveVal(NULL)

    observeEvent(bills(),
        {
            df <- bills()
            if (is.null(df) || nrow(df) == 0) {
                return()
            }

            terms_all <- sort(unique(df$term))
            updatePickerInput(
                session,
                "bill_terms",
                choices = terms_all,
                selected = terms_all
            )
            bm25_obj(bm25_build(df$subject_tokens))
        },
        ignoreInit = TRUE
    )

    bills_scored <- reactive({
        df <- bills()
        if (is.null(df) || nrow(df) == 0) {
            return(tibble())
        }

        # term filter
        terms_all <- sort(unique(df$term))
        sel_terms <- input$bill_terms

        if (!is.null(sel_terms) && length(sel_terms) > 0 && length(sel_terms) < length(terms_all)) {
            df <- df |> filter(term %in% sel_terms)
        }


        # inputs
        kw <- tokenize_simple(input$bill_keyword)
        au <- tokenize_simple(input$bill_author)
        bid <- toupper(gsub("\\s+", "", input$bill_id %||% ""))

        df <- df |> mutate(score = 0)

        any_input <- (nchar(kw) > 0) || (nchar(au) > 0) || (nchar(bid) > 0)

        if (nchar(bid) > 0) {
            df <- df |>
                mutate(
                    id_score = ifelse(str_detect(toupper(gsub("\\s+", "", bill_ID)), fixed(bid)), 1, 0),
                    score = pmax(score, id_score * 1e6)
                )
        }

        if (nchar(au) > 0) {
            qtok <- unique(strsplit(au, " ")[[1]])
            df <- df |>
                mutate(
                    author_score = purrr::map_dbl(.tok, function(x) {
                        if (is.null(x)) {
                            return(0)
                        }
                        if (is.list(x)) x <- unlist(x)
                        xt <- unique(tokenize_simple(paste(x, collapse = " ")))
                        xt <- unique(strsplit(xt, " ")[[1]])
                        if (length(qtok) == 0) {
                            return(0)
                        }
                        length(intersect(qtok, xt)) / max(1, length(qtok))
                    }),
                    score = pmax(score, author_score * 2000)
                )
        }

        if (nchar(kw) > 0) {
            obj <- bm25_obj()
            if (!is.null(obj)) {
                qtok <- unique(strsplit(kw, " ")[[1]])
                df <- df |>
                    mutate(
                        bm25 = purrr::map2_dbl(subject_tokens, obj$dl, ~ bm25_score_one(.x, qtok, obj$idf, obj$avgdl, .y)),
                        score = pmax(score, bm25 * 100)
                    )
            } else {
                df <- df |>
                    mutate(
                        kw_score = ifelse(str_detect(subject_clean, fixed(kw)), 1, 0),
                        score = pmax(score, kw_score * 100)
                    )
            }
        }

        if (!any_input) {
            if ("salience" %in% names(df)) {
                df <- df |> mutate(score = as.numeric(salience %||% 0))
            } else if ("controversy" %in% names(df)) {
                df <- df |> mutate(score = as.numeric(controversy %||% 0))
            } else {
                df <- df |> mutate(score = 0)
            }
        }

        # sorting
        sort_key <- "score"
        df <- df |>
            mutate(
                salience_num = as.numeric(salience %||% NA),
                controversy_num = as.numeric(controversy %||% NA)
            ) |>
            arrange(
                desc(.data[[sort_key]]),
                desc(salience_num %||% 0),
                desc(controversy_num %||% 0)
            )

        n <- input$bill_n %||% 200
        df |>
            distinct() |>
            slice_head(n = n)
    })

    output$bills_dt <- renderDT({
        df <- bills_scored()
        if (nrow(df) == 0) {
            return(datatable(
                tibble(Note = "No results."),
                rownames = FALSE,
                options = list(dom = "t", paging = FALSE),
                class = "display nowrap compact stripe"
            ))
        }

        out <- df |>
            mutate(
                yes_rate = scales::percent(yes_rate_y),
                controv = scales::percent(controversy_pct, scale = 1),
                Outcome = if_else(outcome == 1, "Passed", "Failed"),
                Polarization = scales::percent(polarization, scale = 100)
            ) |>
            transmute(
                `Bill ID` = bill_link,
                Term = term,
                Subject = Subject,
                Authors = Name,
                Outcome,
                `Lifetime (days)` = lifespan_days,
                `Voting Support` = yes_rate,
                `Relative controversy` = controv,
                Polarization
            ) |>
            distinct()

        datatable(
            out,
            rownames = FALSE,
            escape = FALSE,
            extensions = "Buttons",
            options = list(
                pageLength = 15,
                scrollX = TRUE,
                autoWidth = TRUE,
                dom = "<'dt-top dt-top-right'B>t<'dt-bottom'lp>",
                buttons = list(
                    list(extend = "csv", text = "<i class='fa-solid fa-download'></i> Download")
                )
            ),
            class = "display compact stripe"
        )
    })
}

shinyApp(ui, server)
