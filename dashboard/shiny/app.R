library(shiny)
library(stringdist)
library(stringr)
library(shinyWidgets)
library(scales)
library(bslib)
library(DT)
library(arrow)
library(plotly)
library(tidyverse)
library(sf)
library(leaflet)

nyt_theme <- bs_theme(version = 5, base_font = font_google("Open Sans"), heading_font = font_google("Libre Baskerville"), bg = "#ffffff", fg = "#111111", primary = "#000000", secondary = "#555555", base_font_size = "16px", line_height_base = 1.6)

bills <- read_parquet("data/bills.parquet")

reg_funds <- read_csv("data/ca_legislator_funding.csv") %>%
    rename(county_name = NAMELSAD, chamber = house)

reg_topics <- read_csv("data/ca_legislator_topics.csv") %>%
    rename(county_name = NAMELSAD, chamber = house)

county_geoms <- read_sf("data/ca_counties/CA_Counties.shp") %>%
    st_set_crs(3857) %>%
    st_transform(4326) %>%
    rename(county_name = NAMELSAD)

donor_topics <- read_csv("data/donor_lobby_topics.csv")

topics <- read_csv("data/topics_aggregated.csv")

leg_topics <- read_csv("data/legislator_terms.csv")

ui <- fluidPage(
    theme = nyt_theme,
    tags$head(tags$title("Legislative Insights • Beta"), tags$link(rel = "icon", href = "favicon.png")),
    navbarPage(
        title = div(class = "fw-bold", "Legislative Insights"),
        tabPanel(
            "Landing",
            fluidRow(
                column(
                    12,
                    div(class = "card", div(class = "card-header", "Welcome to Legislative Insights"), div(
                        class = "card-body",
                        p("This tool lets you explore California legislative data in a clear, interactive way. You can look at which policies are being proposed, who is writing and supporting them, and where money is coming from. Each section of the app focuses on a different part of that picture. Here's how to use them:"),
                        br(),
                        h5("Map"),
                        p("The map shows how campaign donations and lobbying dollars are distributed across California counties. On the left, select one or more legislative terms (years) and choose which kind of funding to view — donations, lobbying, or both combined. The map will update automatically. You can hover over any county to see detailed numbers, and you can download the full dataset at the bottom of the map."),
                        br(),
                        h5("Search"),
                        p("Use this section to look up specific bills. You can search by bill ID, author name, or a keyword related to the bill's topic. You can also filter by legislative session. The table will show key details about each bill, including when it was introduced, how long it was active, how much support it had, and whether it passed. There's also a link to the full bill text. You can download your search results as a spreadsheet."),
                        br(),
                        h5("Donations & Lobbying"),
                        p("This section lists individual donors and lobbying firms, along with how much they gave and which policy topics they were most involved in. You can choose to view only donors, only lobbyists, or both. The table updates instantly and can be downloaded."),
                        br(),
                        h5("Topics"),
                        p("Here, you can see trends by policy topic. Use the 'Scorecard' tab to choose a topic and view which legislators and funders are most active in that area, along with statistics about bill passage rates, political support, and how controversial the topic tends to be. The 'Data' tab shows a full table of topics across all years and terms, which you can also download."),
                        br(),
                        h5("Legislators"),
                        p("This section shows a list of all legislators by term, along with their main policy areas, how much funding they received, and how many bills they worked on. You can filter, sort, or download the table to learn more about any individual legislator.")
                    ))
                )
            )
        ),
        tabPanel(
            "Map",
            fluidRow(
                column(
                    3,
                    pickerInput(
                        inputId = "term_select",
                        label = "Legislative Term(s)",
                        choices = sort(unique(reg_funds$Term)),
                        selected = unique(reg_funds$Term),
                        multiple = TRUE,
                        options = list(`actions-box` = TRUE, `live-search` = TRUE)
                    ),
                    radioButtons(
                        inputId = "metric_select",
                        label = "Show totals for:",
                        choices = c(
                            "Donations" = "donations",
                            "Lobbying" = "lobbying",
                            "Total" = "total"
                        ),
                        selected = "total",
                        inline = TRUE
                    )
                ),
                column(
                    9,
                    div(
                        class = "card", div(class = "card-header", "Legislative Map"), div(class = "card-body", leafletOutput("countyMap", height = 700)),
                        br(),
                        downloadButton("dl_map_data",
                            "Download Map Data",
                            class = "btn-sm btn-outline-secondary"
                        )
                    )
                )
            )
        ),
        tabPanel(
            "Search",
            fluidRow(
                column(
                    3,
                    div(class = "small text-muted", "Search by Bill ID, Author, or Keyword."),
                    br(),
                    textInput("bill_id", "Bill ID"),
                    pickerInput("session_multi", "Session(s)", choices = sort(unique(bills$term)), multiple = TRUE, width = "fit", options = pickerOptions(countSelectedText = "{0} selected", deselectAllText = "Deselect all", selectAllText = "Select all", selectedTextFormat = "count"), inline = TRUE),
                    textInput("author_text", "Author"),
                    textInput("topic_text", "Keyword"),
                    actionButton("do_search", "Search"),
                    br(),
                    br(),
                    div(class = "small text-muted", "Note: Vote signal represents the proportion of 'yes' votes for the bill. In cases with less voting (i.e. consent calendar), the signal may not reflect actual support."),
                ),
                column(9, div(class = "card", div(class = "card-header", div(class = "d-flex justify-content-between", span("Bills"), downloadButton("dl_bills", "Download CSV", class = "btn-sm btn-outline-secondary"))), div(class = "card-body", DTOutput("bill_tbl"))))
            )
        ),
        tabPanel(
            "Donations & Lobbying",
            fluidRow(
                column(
                    3,
                    radioButtons(
                        inputId = "donor_lobby_select",
                        label = "Show:",
                        choices = c("Donors" = "donor", "Lobbyists" = "lobby_firm", "Both" = "both"),
                        selected = "both",
                        inline = TRUE
                    )
                ),
                column(
                    9,
                    div(
                        class = "card",
                        div(
                            class = "card-header",
                            div(
                                class = "d-flex justify-content-between",
                                span("Donations & Lobbying Table"),
                                downloadButton("dl_donor_lobby", "Download CSV", class = "btn-sm btn-outline-secondary")
                            )
                        ),
                        div(class = "card-body", DTOutput("donor_lobby_tbl"))
                    )
                )
            )
        ),
        tabPanel(
            "Topics",
            tabsetPanel(
                tabPanel(
                    "Scorecard",
                    fluidRow(
                        column(
                            3,
                            uiOutput("topic_select_ui")
                        ),
                        column(
                            9,
                            div(
                                class = "card",
                                div(class = "card-header", "Topic Scorecard"),
                                div(class = "card-body", DTOutput("topic_score_tbl"))
                            )
                        )
                    )
                ),
                tabPanel(
                    "Data",
                    fluidRow(
                        column(
                            12,
                            div(
                                class = "card",
                                div(
                                    class = "card-header d-flex justify-content-between",
                                    span("All Topic Funding Data"),
                                    downloadButton("topic_dl_all", "Download CSV",
                                        class = "btn-sm btn-outline-secondary"
                                    )
                                ),
                                div(class = "card-body", DTOutput("topic_all_tbl"))
                            )
                        )
                    )
                )
            )
        ),
        tabPanel(
            "Legislators",
            fluidRow(
                column(
                    12,
                    div(
                        class = "card",
                        div(
                            class = "card-header d-flex justify-content-between",
                            span("All Legislators"),
                            downloadButton("legis", "Download CSV",
                                class = "btn-sm btn-outline-secondary"
                            )
                        ),
                        div(class = "card-body", DTOutput("leg"))
                    )
                )
            )
        )
    )
)

server <- function(input, output, session) {
    table_opts <- list(pageLength = 25, autoWidth = TRUE, dom = "tp", class = "stripe hover compact nowrap")
    wide_data <- reactive({
        req(input$term_select)
        reg_funds %>%
            filter(Term %in% input$term_select) %>%
            group_by(county_name, chamber) %>%
            summarise(
                donations = sum(donations_total, na.rm = TRUE),
                lobbying = sum(lobbying_total, na.rm = TRUE),
                total = sum(Total, na.rm = TRUE),
            ) %>%
            pivot_wider(
                names_from = chamber,
                values_from = c(donations, lobbying, total),
                names_glue = "{tolower(chamber)}_{.value}"
            ) %>%
            mutate(
                don_sum = rowSums(across(ends_with("_donations")), na.rm = TRUE),
                lob_sum = rowSums(across(ends_with("_lobbying")), na.rm = TRUE),
                tot_sum = rowSums(across(ends_with("_total")), na.rm = TRUE)
            ) %>%
            left_join(county_geoms, by = "county_name") %>%
            st_as_sf()
    })

    output$countyMap <- renderLeaflet({
        df <- wide_data()
        value_col <- switch(input$metric_select,
            donations = "don_sum",
            lobbying = "lob_sum",
            total = "tot_sum"
        )

        pal <- colorQuantile("Blues", domain = df[[value_col]], n = 6)
        assembly_col <- paste0("assembly_", input$metric_select)
        senate_col <- paste0("senate_", input$metric_select)

        labels <- sprintf(
            "<strong>%s</strong><br/>
            Assembly %s: %s<br/>
            Senate&nbsp;&nbsp;%s: %s<br/>
            <u>Combined&nbsp;Total</u>: %s",
            df$county_name,
            tools::toTitleCase(input$metric_select),
            scales::dollar(df[[assembly_col]]),
            tools::toTitleCase(input$metric_select),
            scales::dollar(df[[senate_col]]),
            scales::dollar(df[[value_col]])
        ) |>
            lapply(htmltools::HTML)


        leaflet(df, options = leafletOptions(minZoom = 5, maxZoom = 10)) |>
            addProviderTiles(providers$CartoDB.Positron) |>
            addPolygons(
                fillColor = ~ pal(df[[value_col]]),
                weight = 0.5,
                color = "#666",
                fillOpacity = 0.8,
                label = labels,
                highlightOptions = highlightOptions(
                    weight = 2,
                    color = "#444",
                    fillOpacity = 0.9,
                    bringToFront = TRUE
                )
            ) |>
            addLegend(
                pal = pal,
                values = df[[value_col]],
                title = paste(tools::toTitleCase(input$metric_select), "($)"),
                position = "bottomright",
                labFormat = labelFormat(prefix = "$")
            )
    })

    output$dl_map_data <- downloadHandler(
        filename = function() paste0("county_map_data_", Sys.Date(), ".geojson"),
        content = function(file) {
            sf::st_write(wide_data(), file,
                driver = "GeoJSON", delete_dsn = TRUE
            )
        }
    )

    dtl <- reactive({
        df <- donor_topics %>%
            rename("Area 1" = topic_1, "Area 2" = topic_2, "Area 3" = topic_3, "Area 4" = topic_4, "Area 5" = topic_5, Name = name, Total = total) %>%
            mutate(Total = round(Total, 2))

        if (input$donor_lobby_select == "donor") {
            df <- df %>%
                filter(type == "donor") %>%
                select(-type)
        } else if (input$donor_lobby_select == "lobby_firm") {
            df <- df %>%
                filter(type == "lobby_firm") %>%
                select(-type)
        } else {
            df <- df %>%
                mutate(type = ifelse(type == "donor", "Donor", "Lobbyist")) %>%
                rename(Type = type)
        }
        df %>%
            relocate(`Area 1`:`Area 5`, .after = Total) %>%
            select(-c(topic_0, topic_6, topic_7, topic_8, topic_9))
    })
    output$donor_lobby_tbl <- renderDT({
        datatable(
            dtl(),
            options = list(
                dom = "Blfrtip",
                pageLength = 25,
                autoWidth = TRUE,
                searchHighlight = TRUE,
                class = "stripe hover compact nowrap"
            ),
            rownames = FALSE
        )
    })

    output$dl_donor_lobby <- downloadHandler(
        filename = function() paste0("donor_lobby_", Sys.Date(), ".csv"),
        content  = function(file) write_csv(donor_lobby_res(), file)
    )

    search_res <- eventReactive(input$do_search, {
        df <- bills
        if (length(input$session_multi) > 0) df <- df %>% filter(session %in% input$session_multi)
        if (nchar(trimws(input$author_text)) > 0) {
            q <- str_to_lower(trimws(input$author_text))
            df <- df %>%
                mutate(author_vec = str_split(authors, ",\\s*")) %>%
                mutate(dist_author = map_dbl(author_vec, ~ min(stringdist(q, str_to_lower(str_trim(.x)), method = "jw")))) %>%
                filter(dist_author < 0.4) %>%
                select(-author_vec)
        } else {
            df <- df %>% mutate(dist_author = 0)
        }
        if (nchar(trimws(input$topic_text)) > 0) {
            df <- df %>%
                mutate(dist_topic = stringdist(str_to_lower(C150), str_to_lower(trimws(input$topic_text)), method = "jw")) %>%
                filter(dist_topic < 0.4)
        } else {
            df <- df %>% mutate(dist_topic = 0)
        }
        if (nchar(trimws(input$bill_id)) > 0) {
            id <- trimws(input$bill_id)
            df <- df %>%
                mutate(dist_id = stringdist(bill_id, id, method = "jw")) %>%
                filter(dist_id < 0.3)
        } else {
            df <- df %>% mutate(dist_id = 0)
        }
        df %>%
            arrange(dist_topic, dist_author, dist_id) %>%
            mutate(link = paste0('<a href="https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=', bill_id, '" target="_blank">view</a>')) %>%
            mutate(vote_signal = if_else(vote_signal > 0.5 & outcome == 0, 1 - vote_signal, vote_signal)) %>%
            mutate(
                Introduced = as.Date(First_action),
                Lifespan = paste0(longevity, " days"),
                Topic = C75,
                Outcome = if_else(outcome == 1, "Passed", "Failed"),
                Support = scales::percent(vote_signal, accuracy = 0.1)
            ) %>%
            rename(Bill = bill_id, Authors = authors) %>%
            select(Bill, Introduced, Outcome, Topic, Lifespan, Support, link, Authors)
    })
    output$bill_tbl <- renderDT({
        df <- search_res()
        idx <- which(names(df) == "authors") - 1
        datatable(df, escape = FALSE, rownames = FALSE, options = table_opts) %>%
            formatStyle("Authors", whiteSpace = "normal", wordWrap = "nowrap", lineHeight = "1.2")
    })
    output$dl_bills <- downloadHandler(filename = function() {
        paste0("bills_", Sys.Date(), ".csv")
    }, content = function(file) {
        write_csv(search_res(), file)
    })

    topic_choices <- reactive({
        topics %>%
            distinct(C75) %>%
            filter(C75 != "Acupuncture Licensing and Oversight") %>%
            pull(C75) %>%
            sort()
    })

    output$topic_select_ui <- renderUI({
        selectInput("topic_select", "Topic",
            choices = topic_choices(),
            selected = topic_choices()[1]
        )
    })

    topic_score_data <- reactive({
        req(input$topic_select)

        term_bills <- bills %>% filter(C75 == input$topic_select)
        leg_set <- leg_topics %>% filter(top_0 == input$topic_select)
        ent_set <- donor_topics %>% filter(topic_1 == input$topic_select)

        top_leg <- leg_set %>%
            count(legislator, sort = TRUE) %>%
            head(5) %>%
            pull(legislator)

        top_ent <- ent_set %>%
            rename(entity = name) %>%
            group_by(entity) %>%
            summarise(total = sum(total, na.rm = TRUE)) %>%
            arrange(desc(total)) %>%
            head(5) %>%
            pull(entity)

        passage_rate <- mean(topics %>%
            filter(C75 == input$topic_select) %>%
            pull(outcome))
        r <- leg_topics %>%
            mutate(party = if_else(Party == "D", 1, 0)) %>%
            summarise(party = mean(party, na.rm = TRUE)) %>%
            pull(party)
        part <- (1 - mean(topics %>%
            filter(C75 == input$topic_select) %>%
            pull(partisanship))) / r
        if (is.na(part)) {
            part <- 0
        }
        cont <- 1 - mean(topics %>%
            filter(C75 == input$topic_select) %>%
            pull(vote_signal))


        tibble(
            Metric = c(
                "Top Legislators", "Top Donors/Lobbyists",
                "Passage Rate", "Controversiality", "Bipartisan Support",
                "Volume of Bills"
            ),
            Value = c(
                paste(top_leg, collapse = ", "),
                paste(top_ent, collapse = ", "),
                scales::percent(passage_rate, accuracy = 0.1),
                scales::percent(cont, accuracy = 0.1),
                scales::percent(part, accuracy = 0.1),
                nrow(term_bills)
            )
        )
    })

    output$topic_score_tbl <- renderDT({
        datatable(
            topic_score_data(),
            options = list(dom = "t", paging = FALSE),
            rownames = FALSE
        )
    })

    output$topic_all_tbl <- renderDT({
        datatable(
            topics %>%
                mutate(total_funding = donations_term + lobbying_term) %>%
                filter(C75 != "Acupuncture Licensing and Oversight") %>%
                group_by(C75, term) %>%
                summarise(
                    pass_rate = mean(outcome, na.rm = TRUE),
                    controversiality = mean(vote_signal, na.rm = TRUE),
                    partisanship = mean(partisanship, na.rm = TRUE),
                    total_funding = sum(total_funding, na.rm = TRUE),
                    lobbying = sum(lobbying_term, na.rm = TRUE),
                    donations = sum(donations_term, na.rm = TRUE),
                    number_bills = sum(bill_id, na.rm = TRUE),
                    longevity = mean(longevity, na.rm = TRUE)
                ) %>%
                rename(Topic = C75, Term = term) %>%
                mutate(
                    pass_rate = scales::percent(pass_rate, accuracy = 0.1),
                    controversiality = scales::percent(1 - controversiality, accuracy = 0.1),
                    partisanship = if_else(partisanship < 0.7, paste0(scales::percent((0.7 - partisanship) / 0.7, accuracy = 0.1), " R."), if_else(partisanship > 0.7, paste0(scales::percent((partisanship - 0.7) / 0.3, accuracy = 0.1), " D."), "Neutral")),
                    total_funding = scales::dollar(total_funding, accuracy = 1),
                    lobbying = scales::dollar(lobbying, accuracy = 1),
                    donations = scales::dollar(donations, accuracy = 1)
                ),
            options = list(
                dom = "Blfrtip",
                pageLength = 15,
                autoWidth = TRUE,
                searchHighlight = TRUE
            ),
            rownames = FALSE
        )
    })

    output$topic_dl_all <- downloadHandler(
        filename = function() paste0("topic_funding_all_", Sys.Date(), ".csv"),
        content  = function(file) write_csv(donor_topics, file)
    )

    leg_tbl <- reactive({
        leg_topics %>%
            mutate(
                total_funding = donations_total + lobbying_total,
                lobbying_term = scales::dollar(lobbying_total, accuracy = 1),
                donations_term = scales::dollar(donations_total, accuracy = 1),
                total_funding = scales::dollar(total_funding, accuracy = 1),
                outcome = scales::percent(outcome, accuracy = 0.1)
            ) %>%
            rename(Legislator = legislator, Term = term, House = chamber, "Area 1" = top_0, "Area 2" = top_1, "Area 3" = top_2, "Area 4" = top_3, "Area 5" = top_4, "Area 6" = top_5, "Area 7" = top_6, "Area 8" = top_7, "Area 9" = top_8, "Area 10" = top_9, Outcome = outcome, "Lead Author" = author_type, "Number of Bills" = bill_version, Donations = donations_term, Lobbying = lobbying_term, "Total Funding" = total_funding) %>%
            select(Legislator, Term, House, Outcome, "Lead Author", "Number of Bills", Donations, Lobbying, "Total Funding", "Area 1":"Area 10")
    })

    output$leg <- renderDT({
        datatable(leg_tbl(),
            options = list(
                dom = "Blfrtip",
                pageLength = 25,
                autoWidth = TRUE,
                searchHighlight = TRUE,
                class = "stripe hover compact nowrap"
            )
        )
    })

    output$legis <- downloadHandler(
        filename = function() {
            paste0("legislators_", Sys.Date(), ".csv")
        },
        content = function(file) {
            write_csv(leg_tbl(), file)
        }
    )
}

shinyApp(ui, server)
