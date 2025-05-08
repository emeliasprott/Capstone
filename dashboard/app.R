if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
    shiny, bslib, arrow, data.table, dplyr, plotly, DT, lubridate,
    reticulate, fontawesome, htmltools, scales
)
reticulate::use_condaenv("faiss", required = TRUE)

# ======================================================
# preprocessing
# ======================================================
dpath <- "backend/data"
bill <- read_parquet(file.path(dpath, "bill.parquet")) %>% as.data.table()
leg <- read_parquet(file.path(dpath, "legislator_term.parquet")) %>% as.data.table()
committee <- read_parquet(file.path(dpath, "committee.parquet")) %>% as.data.table()
donor <- read_parquet(file.path(dpath, "donor.parquet")) %>% as.data.table()
lobby <- read_parquet(file.path(dpath, "lobby_firm.parquet")) %>% as.data.table()
topics <- read_parquet(file.path(dpath, "topic_summary.parquet")) %>% as.data.table()
policy <- read_parquet(file.path(dpath, "policy_session.parquet")) %>% as.data.table()
policy_topics <- c(
    "K-12 Education",
    "Public Universities",
    "Technical Education & Job Readiness",
    "Affordable Housing",
    "Tenants\' Rights",
    "Homelessness",
    "Drought Management",
    "Wildfire Prevention",
    "Electricity Grid Reliability",
    "Energy Efficiency",
    "Air Quality",
    "Sea Level Rise & Coastal Resilience",
    "Public Transit Infrastructure",
    "Highways & Road Maintenance",
    "Broadband Internet Access",
    "Mental Health Services & Crisis Intervention",
    "Substance Use Disorder & Harm Reduction",
    "Child Welfare & Foster Care",
    "Elder Care & Aging Services",
    "Disability Rights & Accessibility",
    "Healthcare Access & Medi-Cal",
    "Public Health & Disease Control",
    "Food Insecurity & Nutrition Assistance",
    "Environmental Justice",
    "Water Pollution",
    "Agricultural Regulation",
    "Coastal Protection",
    "Parks and Public Lands",
    "Criminal Justice Reform",
    "Police Accountability",
    "Firearm Regulations",
    "Emergency Management & Disaster Response",
    "Human Trafficking Prevention",
    "Budget Reserves & Fiscal Stabilization",
    "Local Government Finance & Property Taxes",
    "State Tax Policy",
    "Cannabis Regulation",
    "Insurance Oversight & Consumer Protection",
    "Small Business Development",
    "Technology Regulation",
    "Political Transparency",
    "Voting Rights & Election Security",
    "Government Transparency & Public Records",
    "Immigration Protections & Reform",
    "Gender Equity & Reproductive Rights",
    "LGBTQ+ Rights",
    "Racial Equity & Anti-Discrimination",
    "Veterans Services & Support",
    "Labor Rights & Minimum Wage",
    "Paid Leave",
    "Workforce Development & Job Training",
    "State Employee Pensions & Public Retirement",
    "Public Transportation Safety & Accessibility",
    "Affordable Childcare",
    "Consumer Protection",
    "Fair Housing",
    "Wildlife Conservation & Endangered Species",
    "Renewable Energy",
    "Natural Gas Regulation",
    "Vehicle Emissions",
    "Oil Drilling & Fracking",
    "Tribal Affairs",
    "Military & Veterans Affairs",
    "Public Health & Disease Control",
    "Information Technology",
    "Prisons & Corrections",
    "Child Support",
    "Public Libraries",
    "Utilities Oversight",
    "Regional Investment & Job Creation",
    "Public Employee Relations",
    "Manufactured Housing Tenant Protections & Park Regulations",
    "Short Term Rental Regulation",
    "Public Utility Wildfire Mitigation",
    "Energy Storage",
    "Water Recycling",
    "Urban Heat Island Mitigation",
    "Stormwater Capture",
    "Salton Sea Restoration",
    "Public Employee Health Benefits",
    "State Procurement of Goods & Services",
    "Correctional Officer Rights",
    "Traffic Enforcement",
    "Public Charter Schools",
    "Ethnic Studies Curriculum",
    "Library Construction",
    "Toxic Chemical Disclosure",
    "Warehouse Labor Standards",
    "Worker Classification & Independent Contractors",
    "Flood Control",
    "Unclaimed Property",
    "Alcohol Beverage Control",
    "Youth Diversion Programs",
    "Sexual Harassment Prevention",
    "Unemployment Insurance",
    "Sea Level Rise Adaptation"
)

label_map <- data.table(topic_id = seq_along(policy_topics) - 1L, topic = policy_topics)

bill <- merge(bill, label_map, by.x = "dominant_topic", by.y = "topic_id", all.x = TRUE)

topics <- merge(topics, label_map, by.x = "topic", by.y = "topic_id", all.x = TRUE)

add_topic_label <- function(dt, id_col = "top_topic") {
    if (id_col %in% names(dt)) {
        merge(dt, label_map, by.x = id_col, by.y = "topic_id", all.x = TRUE)
    } else {
        dt
    }
}
leg <- add_topic_label(leg)
committee <- add_topic_label(committee)
donor <- add_topic_label(donor)
lobby <- add_topic_label(lobby)

faiss <- import("faiss", delay_load = TRUE)
pickle <- import("pickle", delay_load = TRUE)
builtins <- import_builtins()
idx <- faiss$read_index(file.path("backend/data", "bill_sim.faiss"))
f <- builtins$open(file.path("backend/data", "bill_sim_ids.pkl"), "rb")
id_map <- pickle$load(f)
f$close()
emb_files <- list.files("backend/data", "bill_embeddings_.*\\.parquet", full.names = TRUE)
emb_ds <- open_dataset(emb_files)
get_vec <- function(bid) {
    row <- emb_ds %>%
        filter(node_id == bid) %>%
        collect()
    if (nrow(row) == 0) {
        return(NULL)
    }
    unlist(row$embedding[[1]])
}
nearest <- function(bid, k = 10) {
    v <- get_vec(bid)
    if (is.null(v)) {
        return(integer(0))
    }
    v <- v / sqrt(sum(v * v))
    I <- idx$search(matrix(v, nrow = 1), k)[[2]][1, ] + 1L
    id_map[I]
}

topics[, avg_success := round(avg_success * 100, 2)]
topics[, avg_polar := round(avg_polar * 100, 2)]

recent_cut <- max(bill$intro_date) - 90
recent <- bill[intro_date >= recent_cut]

# ======================================================
# UI
# ======================================================
theme <- bs_theme(
    base_font = font_google("Open Sans"),
    heading_font = font_google("Inter"),
    primary = "#0061A8",
    secondary = "#4FB3BF",
    "body-bg" = "#f4f7fa",
    "card-bg" = "rgba(255,255,255,.55)",
    "card-border-width" = "0"
)
theme <- bs_add_rules(theme, "
.card-glass{
  backdrop-filter:blur(12px) saturate(180%);
  -webkit-backdrop-filter:blur(12px) saturate(180%);
  box-shadow:0 8px 18px rgba(0,0,0,.07)!important;
  border-radius:1rem!important;
  transition:transform .3s ease;
}
.card-glass:hover{ transform:translateY(-5px); }
.btn-pill{ border-radius:9999px!important; }
")

ui <- navbarPage(
    title = tags$span(class = "fw-bold text-uppercase small", "California Legislative Monitor"),
    theme = theme, id = "nav",

    # -------- Overview ------------------------------------------------------
    tabPanel(
        "Overview",
        div(
            class = "container-xl py-5",
            h1(class = "display-5 fw-bold text-primary mb-4", "Legislative Snapshot"),
            h3(class = "mt-5 mb-3", "Top Topics by Bill Count"),
            {
                topN <- topics[order(-n_bills)][1:10]
                plot_ly(topN,
                    x = ~ reorder(topic, n_bills), y = ~n_bills, type = "bar",
                    marker = list(color = "#4FB3BF")
                ) |>
                    layout(
                        xaxis = list(title = "", tickangle = -45),
                        yaxis = list(title = "Bills")
                    )
            }
        )
    ),

    # -------- Trends --------------------------------------------------------
    tabPanel(
        "Trends",
        div(
            class = "container-xl py-5",
            h2("Bills & Funding by Session"),
            selectInput("tr_topic", NULL,
                choices = sort(unique(policy$topic)),
                selected = head(sort(unique(policy$topic)), 3),
                multiple = TRUE, class = "form-select btn-pill mb-4 fs-6"
            ),
            plotlyOutput("trend_plot", height = 450)
        )
    ),

    # -------- Topic ---------------------------------------------------------
    tabPanel(
        "Topic",
        div(
            class = "container-xl py-5",
            selectInput("ex_topic", NULL,
                choices = sort(unique(topics$topic)),
                class = "form-select btn-pill fs-5 mb-4"
            ),
            h4("Sessions"),
            plotlyOutput("topic_plot", height = 300),
            hr(class = "my-4"),
            h4("Top Actors"),
            DTOutput("actor_tbl"),
            hr(class = "my-4"),
            h4("Recent Bills"),
            DTOutput("recent_tbl")
        )
    ),

    # -------- Search --------------------------------------------------------
    tabPanel(
        "Search",
        div(
            class = "container-xl py-5",
            textInput("kw", NULL,
                placeholder = "keyword in title",
                class = "form-control w-50 d-inline"
            ),
            actionButton("btn_kw", tagList(icon("magnifying-glass"), "Search"),
                class = "btn btn-primary btn-pill ms-2 px-4"
            ),
            hr(),
            DTOutput("kw_results"),
            hr(class = "my-5"),
            h3("Find Similar Bills"),
            numericInput("sim_id", "Bill ID", 1, min = 1, width = "25%"),
            actionButton("btn_sim", tagList(icon("sparkles"), "Find"),
                class = "btn btn-secondary btn-pill"
            ),
            hr(),
            DTOutput("sim_results")
        )
    )
)

# ======================================================
# Server
# ======================================================
server <- function(input, output, session) {
    # ---- Trends -----------------------------------------------------------
    output$trend_plot <- renderPlotly({
        req(input$tr_topic)
        df <- policy[topic %in% input$tr_topic]
        pal <- RColorBrewer::brewer.pal(8, "Set2")
        p <- plot_ly()
        i <- 1
        for (t in input$tr_topic) {
            sub <- df[topic == t]
            p <- add_lines(p,
                x = sub$session, y = sub$n_bills,
                name = paste(t, "Bills"), line = list(color = pal[i], width = 2)
            )
            p <- add_lines(p,
                x = sub$session, y = sub$total_funding / 1e6,
                name = paste(t, "Funding $M"), yaxis = "y2",
                line = list(color = pal[i], dash = "dot")
            )
            i <- i + 1
        }
        layout(p, yaxis2 = list(
            title = "$ Millions",
            overlaying = "y", side = "right"
        ))
    })

    # ---- Topic KPIs ------------------------------------------------------
    tp_row <- reactive(topics[topic == input$ex_topic])

    lapply(c("n_bills", "total_dollars", "avg_success", "avg_polar"), \(col){
        output[[paste0("kpi_", col)]] <- renderText({
            v <- tp_row()[[col]]
            switch(col,
                total_dollars = dollar(v),
                avg_success   = percent(v),
                avg_polar     = percent(v),
                v
            )
        })
    })

    output$topic_plot <- renderPlotly({
        sub <- policy[topic == input$ex_topic]
        plot_ly(sub, x = ~session) |>
            add_bars(y = ~n_bills, name = "Bills", marker = list(color = "#168aad")) |>
            add_lines(
                y = ~ total_funding / 1e6, name = "Funding $M",
                line = list(color = "#f49f0a", width = 2), yaxis = "y2"
            ) |>
            layout(yaxis2 = list(overlaying = "y", side = "right", title = "$ Millions"))
    })


    output$recent_tbl <- DT::renderDT(
        {
            recent[topic == input$ex_topic][order(-intro_date)][
                , .(intro_date, bill_id, title, success_risk, polarisation_score)
            ]
        },
        options = list(pageLength = 8, scrollX = TRUE)
    )

    # ---- Keyword search --------------------------------------------------
    observeEvent(input$btn_kw, {
        kw <- tolower(trimws(input$kw))
        res <- if (nchar(kw)) {
            bill[
                grepl(kw, tolower(title)),
                .(
                    intro_date, bill_id, title, topic,
                    success_risk, polarisation_score
                )
            ]
        } else {
            data.table()
        }
        output$kw_results <- DT::renderDT(res,
            options = list(pageLength = 12, scrollX = TRUE)
        )
    })

    # ---------- SEARCH ----------------------------------
    observeEvent(input$btn_find, {
        ids <- nearest(input$search_billId, k = 10)
        df <- rv()$bill[bill_id %in% ids]
        output$tbl_similar <- renderDT(datatable(df, options = list(scrollX = TRUE)))

        main <- rv()$bill[bill_id == input$search_billId]
        if (nrow(main)) {
            output$txt_bill_profile <- renderText({
                sprintf(
                    "Bill %s • Topic: %s • Success Risk %.2f • Polarisation %.2f",
                    main$bill_id, main$topic, main$success_risk, main$polarisation_score
                )
            })
        }
    })
}

shinyApp(ui = ui, server = server)
