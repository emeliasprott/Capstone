# ======================================================
# preprocessing
# ======================================================
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
    shiny, bslib, arrow, data.table, dplyr, plotly, DT, lubridate,
    reticulate, scales, RColorBrewer
)

reticulate::use_condaenv("faiss", required = FALSE)
faiss <- reticulate::import("faiss", delay_load = TRUE)
pickle <- reticulate::import("pickle", delay_load = TRUE)
builtins <- reticulate::import_builtins()

dpath <- "backend/data"
safe <- \(f) tryCatch(read_parquet(f) |> as.data.table(), error = \(e) data.table())

bill <- safe(file.path(dpath, "bill.parquet"))
leg <- safe(file.path(dpath, "legislator_term.parquet"))
committee <- safe(file.path(dpath, "committee.parquet"))
donor <- safe(file.path(dpath, "donor.parquet"))
lobby <- safe(file.path(dpath, "lobby_firm.parquet"))
topics <- safe(file.path(dpath, "topic_summary.parquet"))
policy <- safe(file.path(dpath, "policy_session.parquet"))

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

idx_file <- file.path(dpath, "bill_sim.faiss")
pkl_file <- file.path(dpath, "bill_sim_ids.pkl")
emb_files <- list.files(dpath, "bill_embeddings_.*\\.parquet", full.names = TRUE)

if (file.exists(idx_file) && file.exists(pkl_file) && length(emb_files)) {
    idx <- faiss$read_index(idx_file)
    id_map <- pickle$load(builtins$open(pkl_file, "rb"))
    emb_ds <- open_dataset(emb_files)

    get_vec <- \(bid){
        row <- emb_ds %>%
            filter(node_id == bid) %>%
            collect()
        if (!nrow(row)) NULL else unlist(row$embedding[[1]])
    }
    nearest <- \(bid, k = 10){
        v <- get_vec(bid)
        if (is.null(v)) {
            return(integer(0))
        }
        v <- v / sqrt(sum(v * v))
        idx$search(matrix(v, 1), k)[[2]][1, ] + 1L |> (\(I) id_map[I])()
    }
} else {
    nearest <- \(...) integer(0)
}
topics[, `:=`(
    success_pct = round(avg_success * 100, 2),
    polar_pct = round(avg_polar * 100, 2)
)]
bill <- bill %>%
    mutate(outcome = if_else(outcome > 0, 1, 0))
avg_outcome <- bill %>%
    group_by(topic) %>%
    summarise(avg_outcome = mean(outcome, na.rm = TRUE))
nbills_per_topic <- bill %>%
    group_by(topic) %>%
    summarise(nbills = n()) %>%
    ungroup()

topics <- topics %>%
    left_join(nbills_per_topic, by = "topic")

topics$avg_outcome <- avg_outcome$avg_outcome[match(topics$topic, avg_outcome$topic)]
topics$success_pct <- round(topics$avg_outcome * 100, 2)

contro <- scale(topics$polar_pct, center = TRUE, scale = TRUE)
topics$controversiality <- (contro + 2) * 1.25
# ======================================================
# UI
# ======================================================
theme <- bs_theme(
    version = 5,
    base_font = font_google("Open Sans"),
    heading_font = font_google("Inter"),
    primary = "#004E89",
    secondary = "#A7C957",
    "body-bg" = "#f4f7fa"
)

vbox <- function(title, value, subtitle = NULL, icon = NULL, color = NULL) {
    bslib::value_box(
        title = title, value = value, showcase = icon,
        footer = subtitle, theme_color = color %||% "secondary"
    )
}

ui <- navbarPage(
    title = "California Legislative Monitor", theme = theme, id = "nav",
    tabPanel(
        "Overview",
        div(
            class = "container-xl py-4",
            h3("Overview of Policy Areas"),
            p(
                "Controversiality is a measure of how much support and opposition a topic receives in the legislature, compared to other topics. On this scale, 0 is the least controversial and 5 is the most controversial.",
                style = "font-size: 12px; margin-top: 10px;"
            ),
            plotlyOutput("bubble_topics", height = 400)
        ),
        div(
            class = "container-xl py-4",
            h3("Funding and Lobbying Overview"),
            fluidRow(
                column(
                    3,
                    selectInput(
                        "metric", "Choose Metric:",
                        choices = c(
                            "Total Bills Authored" = "num_authored_bills",
                            "Percent Bills Passed" = "pct_passed",
                            "Influence Score" = "leverage"
                        ),
                        selected = "leverage"
                    )
                ),
                column(
                    9,
                    plotlyOutput("leg_metrics_bar", height = "500px")
                )
            )
        )
    ),
    tabPanel(
        "Bill search",
        div(
            class = "container-xl py-4",
            textInput("kw", "Keyword in title"),
            actionButton("btn_kw", "Search"),
            DTOutput("kw_results"), hr(),
            numericInput("search_billId", "Bill ID", min = min(bill$bill_id), value = head(bill$bill_id, 1)),
            actionButton("btn_find", "Find similar"),
            DTOutput("tbl_similar")
        )
    )
)

# ======================================================
# Server
# ======================================================
server <- function(input, output, session) {
    output$bubble_topics <- renderPlotly({
        plot_ly(
            topics,
            x = ~controversiality, y = ~ avg_outcome * 100, type = "scatter",
            mode = "markers",
            size = 45,
            text = ~topic,
            hovertemplate = "<b>%{text}</b><br>Chaptered % %{y:.2f}<br>Polarisation % %{x:.2f}",
            marker = list(color = "#A7C957", opacity = .7, line = list(width = 1, color = "#004E89"))
        ) |> layout(
            xaxis = list(title = "Relative Controversiality<br><sup>(On a scale from 0 to 5)</sup>"),
            yaxis = list(title = "Successful legislation (%)")
        )
    })
    output$leg_metrics_bar <- renderPlotly({
        req(input$metric)
        dt <- leg %>%
            filter(!is.na(topic_focus_y))

        if (input$metric == "leverage") {
            dt <- dt %>%
                rename(
                    "Topic Focus" = topic_focus_y
                )
            ggplotly(
                ggplot(dt, aes(x = !!sym(input$metric), y = `Topic Focus`, fill = `Topic Focus`, text = name)) +
                    annotate(
                        "rect",
                        xmin = -1e-4, xmax = 0, ymin = 0, ymax = 18,
                        fill = "red", alpha = 0.15
                    ) +
                    annotate(
                        "rect",
                        xmin = 0, xmax = 1e-4, ymin = 0, ymax = 18,
                        fill = "green", alpha = 0.15
                    ) +
                    scale_x_continuous(
                        limits = c(-1e-4, 1e-4)
                    ) +
                    geom_jitter() +
                    guides(fill = "none") +
                    labs(x = input$metric, y = "Topic Focus") +
                    theme_minimal() +
                    theme(
                        axis.title.y = element_blank(),
                        axis.text.y = element_text(size = 8, margin = margin(r = 0, l = 0))
                    ),
                tooltip = c("text", "x", "y"),
                hovertemplate = "<b>%{text}</b><br>%{x:.2f} %{y}"
            )
        } else if (input$metric == "num_authored_bills") {
            ggplotly(
                ggplot(dt, aes(x = !!sym(input$metric), y = topic_focus_y, fill = topic_focus_y, text = name)) +
                    geom_jitter() +
                    guides(fill = "none") +
                    labs(x = input$metric, y = "Topic Focus") +
                    theme_minimal() +
                    theme(
                        axis.title.y = element_blank(),
                        axis.text.y = element_text(size = 8, margin = margin(r = 0, l = 0))
                    ),
                tooltip = c("text", "x", "y"),
                hovertemplate = "<b>%{text}</b><br>%{x:.2f} %{y}"
            )
        } else {
            dt <- dt %>%
                mutate(
                    pct_passed = if_else(
                        is.na(as.numeric(pct_passed)),
                        0,
                        as.numeric(pct_passed) * 100
                    )
                )
            ggplotly(
                ggplot(dt, aes(x = pct_passed, y = topic_focus_y, fill = topic_focus_y, text = name)) +
                    geom_jitter() +
                    guides(fill = "none") +
                    labs(x = "% Passed", y = "Topic Focus") +
                    theme_minimal() +
                    theme(
                        axis.title.y = element_blank(),
                        axis.text.y = element_text(size = 8, margin = margin(r = 0, l = 0))
                    ),
                tooltip = c("text", "x", "y"),
                hovertemplate = "<b>%{text}</b><br>%{x} %{y}"
            )
        }
    })

    observeEvent(input$btn_kw, {
        kw <- tolower(trimws(input$kw))
        res <- if (nzchar(kw)) {
            bill[
                grepl(kw, tolower(title)),
                .(bill_id, title, topic, success_risk, polarisation_score)
            ]
        } else {
            data.table()
        }
        output$kw_results <- DT::renderDT(res, options = list(pageLength = 15, scrollX = TRUE))
    })

    observeEvent(input$btn_find, {
        ids <- nearest(input$search_billId, 10)
        output$tbl_similar <- DT::renderDT(
            bill[bill_id %in% ids, .(bill_id, title, topic, success_risk, polarisation_score)],
            options = list(pageLength = 10, scrollX = TRUE)
        )
    })
}

shinyApp(ui = ui, server = server)
