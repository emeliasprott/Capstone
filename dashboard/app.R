if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
    shiny, shinydashboard, arrow, data.table, dplyr, plotly,
    ggplot2, DT, lubridate, networkD3, reticulate, htmltools
)
reticulate::use_condaenv("faiss", required = TRUE)

# ======================================================
# preprocessing
# ======================================================
dpath <- "backend/data"
bill <- read_parquet(file.path(dpath, "bill_kpis.parquet")) %>% as.data.table()
leg <- read_parquet(file.path(dpath, "legislator_kpis.parquet")) %>% as.data.table()
committee <- read_parquet(file.path(dpath, "committee_kpis.parquet")) %>% as.data.table()
donor <- read_parquet(file.path(dpath, "donor_kpis.parquet")) %>% as.data.table()
lobby <- read_parquet(file.path(dpath, "lobby_firm_kpis.parquet")) %>% as.data.table()
topic_sn <- read_parquet(file.path(dpath, "topic_snapshot.parquet")) %>% as.data.table()
policy_topics <- c(
    "K-12 school finance teacher retention student performance education equity",
    "California public universities tuition affordability financial aid access higher education funding",
    "Career technical education workforce training vocational pathways job readiness programs",
    "Affordable housing development zoning density bonus housing supply homelessness prevention",
    "Tenant rights eviction protections rent control rental housing affordability",
    "Homelessness supportive housing unhoused populations shelter access wraparound services",
    "Water rights drought management agricultural water use groundwater sustainability urban conservation",
    "Wildfire prevention forest management defensible space fire suppression vegetation clearance",
    "Electricity grid reliability utility regulation wildfire liability renewable energy integration",
    "Decarbonization energy efficiency building electrification appliance standards clean energy transition",
    "Air quality emissions standards pollution control disadvantaged community mitigation",
    "Climate adaptation sea level rise coastal resilience flood control urban heat islands",
    "Public transit infrastructure rail expansion zero emission buses traffic congestion reduction",
    "Highways road maintenance state infrastructure funding transportation improvement projects",
    "Broadband internet access rural connectivity digital equity infrastructure deployment grants",
    "Mental health services crisis intervention psychiatric care Medi-Cal behavioral health",
    "Substance use disorder harm reduction opioid treatment fentanyl prevention drug diversion",
    "Child welfare foster care adoption permanency social worker caseload family reunification",
    "Elder care long term services IHSS caregiver workforce nursing home oversight",
    "Disability rights accessibility independent living ADA compliance supportive services",
    "Healthcare access Medi-Cal managed care provider reimbursement hospital funding",
    "Public health disease control immunizations pandemic preparedness local health departments",
    "Food insecurity CalFresh food banks nutrition assistance hunger prevention programs",
    "Environmental justice pollution burdens cumulative impacts disadvantaged community protections",
    "Water pollution stormwater runoff drinking water contaminants safe clean supply",
    "Agricultural regulation pesticide use farmworker protections sustainable agriculture soil conservation",
    "Coastal protection shoreline preservation habitat restoration ocean pollution prevention",
    "Parks and public lands recreation trail access conservation funding habitat protection",
    "Criminal justice reform sentencing reduction probation diversion incarceration alternatives",
    "Police accountability body cameras use of force civilian oversight public trust",
    "Firearm regulation background checks red flag laws gun violence prevention",
    "Emergency management disaster response wildfire preparedness flood recovery coordination",
    "Human trafficking prevention victim services law enforcement anti-trafficking task forces",
    "Budget reserves rainy day fund fiscal stabilization bond debt management",
    "Local government finance property taxes redevelopment agencies municipal fiscal health",
    "State tax policy corporate taxes income taxes sales tax exemptions revenue generation",
    "Cannabis regulation retail licensing cultivation taxation youth access prevention",
    "Insurance oversight consumer protection health auto homeowner premium rate review",
    "Small business development regulatory relief entrepreneurship minority business enterprise support",
    "Technology regulation consumer privacy data security artificial intelligence algorithmic accountability",
    "Campaign finance disclosure independent expenditures lobbying ethics political transparency",
    "Voting rights election security voter registration mail ballot access ranked choice voting",
    "Government transparency public records Brown Act open meetings whistleblower protections",
    "Immigration protections undocumented residents sanctuary policies deportation defense",
    "Gender equity reproductive rights workplace discrimination equal pay parental leave",
    "LGBTQ+ rights transgender protections school inclusion hate crimes prevention",
    "Racial equity anti-discrimination restorative justice inclusive education curriculum reform",
    "Veterans services housing benefits employment reintegration mental health supports",
    "Labor rights minimum wage wage theft enforcement collective bargaining workplace safety",
    "Paid leave family leave sick leave medical leave small business impacts",
    "Workforce development apprenticeships job training reskilling economic mobility programs",
    "State employee pensions public retirement systems CalPERS benefit reform",
    "Public transportation safety accessibility fare affordability low income rider assistance",
    "Affordable childcare early learning preschool provider workforce child development programs",
    "Consumer protection financial scams fraud prevention debt collection predatory lending",
    "Housing discrimination fair housing enforcement zoning segregation equitable development",
    "Wildlife conservation endangered species habitat preservation biodiversity protection programs",
    "Renewable energy solar net metering wind energy grid integration clean power",
    "Natural gas regulation methane emissions infrastructure safety gas appliance standards",
    "Vehicle emissions zero emission vehicle incentives smog checks air pollution",
    "Oil drilling fracking regulation offshore oil platform decommissioning environmental safety",
    "Tribal affairs land sovereignty gaming compacts economic development native representation",
    "Military and veterans affairs state military department national guard emergency response",
    "COVID-19 pandemic response public health emergency vaccination testing mitigation efforts",
    "State information technology cybersecurity modernization digital services user access equity",
    "Corrections prison population rehabilitation reentry parole community supervision",
    "Child support enforcement family court collections paternity establishment financial obligations",
    "Public libraries literacy programs broadband access educational resources community hubs",
    "Utilities oversight CPUC rate setting electricity gas telecommunications public accountability",
    "Economic development regional investment job creation economic stimulus small business loans",
    "Public employee relations collective bargaining strikes grievance arbitration workforce disputes",
    "Mobile home rent stabilization manufactured housing tenant protections park regulations",
    "Short term rental regulation vacation housing zoning neighborhood impacts tourist displacement",
    "Public utility wildfire mitigation vegetation management electrical infrastructure safety plans",
    "Energy storage battery incentive programs peak load management renewable energy integration",
    "Water recycling wastewater reuse potable reuse treatment groundwater replenishment",
    "Urban heat island mitigation tree canopy expansion cool pavements building codes",
    "Stormwater capture green infrastructure runoff management urban flood prevention",
    "Salton Sea restoration air quality dust mitigation habitat conservation regional collaboration",
    "Public employee health benefits bargaining premium contributions retiree medical obligations",
    "State procurement diversity small business disadvantaged business enterprise contracting goals",
    "Correctional officer labor relations staffing levels overtime prison safety workplace rights",
    "Traffic enforcement automated speed cameras red light violations pedestrian safety programs",
    "Public charter school accountability fiscal oversight renewal and revocation standards",
    "Ethnic studies curriculum implementation graduation requirement culturally responsive pedagogy",
    "Library construction bond funding capital outlay public library infrastructure modernization",
    "Toxic chemical disclosure Proposition 65 safe harbor levels consumer product safety",
    "Warehouse labor standards workplace quotas ergonomic protections injury prevention policies",
    "Gig economy worker classification independent contractor disputes AB 5 exemptions",
    "Flood control levee maintenance state-federal project funding Delta and Central Valley protection",
    "Unclaimed property escheatment holder compliance reunification outreach owner claims process",
    "Alcohol beverage control licensing tied house restrictions responsible beverage service training",
    "Youth diversion programs juvenile delinquency prevention community-based alternatives criminal penalties",
    "Sexual harassment prevention workplace training employer liability legal compliance",
    "Unemployment insurance trust fund solvency employer tax rates benefit payment delays",
    "Sea level rise adaptation coastal erosion infrastructure relocation living shoreline projects"
)

label_map <- data.table(topic_id = seq_along(policy_topics) - 1L, topic = policy_topics)

bill <- merge(bill, label_map, by.x = "dominant_topic", by.y = "topic_id", all.x = TRUE)

topic_sn <- merge(topic_sn, label_map, by = "topic_id", all.x = TRUE)

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

setnames(leg, old = "topic", new = "top_topic_label", skip_absent = TRUE)
setnames(committee, old = "topic", new = "top_topic_label", skip_absent = TRUE)
setnames(donor, old = "topic", new = "top_topic_label", skip_absent = TRUE)
setnames(lobby, old = "topic", new = "top_topic_label", skip_absent = TRUE)

make_actor_tbl <- function(dt, type, possible_namecols) {
    name_col <- possible_namecols[possible_namecols %in% names(dt)][1]
    if (is.null(name_col) || is.na(name_col)) {
        dt[, actor := as.character(node_id)]
    } else {
        dt[, actor := get(name_col)]
    }
    need <- c("topic_focus", "influence", "leverage", "bipartisan_score", "top_topic_label")
    for (n in need[!need %in% names(dt)]) dt[, (n) := NA]
    dt[, .(node_id, actor, type = type, top_topic_label, topic_focus, influence, leverage, bipartisan_score)]
}

actors <- rbindlist(list(
    make_actor_tbl(leg, "Legislator", c("legislator_name", "name")),
    make_actor_tbl(committee, "Committee", c("committee_name", "name")),
    make_actor_tbl(donor, "Donor", c("donor_name", "name")),
    make_actor_tbl(lobby, "Lobby Firm", c("firm_name", "name"))
), fill = TRUE)

setorder(actors, -influence, na.last = TRUE)

topic_summary <- bill[, .(
    n_bills        = .N,
    total_funding  = sum(lobbying_amt_sum, na.rm = TRUE),
    avg_success    = mean(success_risk, na.rm = TRUE),
    avg_polar      = mean(polarisation_score, na.rm = TRUE)
), by = topic]

faiss <- import("faiss", delay_load = TRUE)
pickle <- import("pickle", delay_load = TRUE)
builtins <- import_builtins()
idx <- faiss$read_index(file.path(dpath, "bill_sim.faiss"))
f <- builtins$open(file.path(dpath, "bill_sim_ids.pkl"), "rb")
id_map <- pickle$load(f)
f$close()
emb_files <- list.files(dpath, "bill_embeddings_.*\\.parquet", full.names = TRUE)
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
topic_summary <- bill[, .(n_bills = .N, total_funding = sum(lobbying_amt_sum, na.rm = TRUE), avg_success = mean(success_risk, na.rm = TRUE), avg_polar = mean(polarisation_score, na.rm = TRUE)), by = topic]


# ======================================================
# UI
# ======================================================
header <- dashboardHeader(title = "California Legislative Intelligence")

sidebar <- dashboardSidebar(
    sidebarMenu(
        menuItem("Overview", tabName = "overview", icon = icon("dashboard")),
        menuItem("Topic Dashboard", tabName = "topic_tr", icon = icon("layer-group")),
        menuItem("Actors", tabName = "actors", icon = icon("users")),
        menuItem("Risk vs Polarisation", tabName = "risk", icon = icon("braille")),
        menuItem("Funding Map", tabName = "fund_map", icon = icon("hand-holding-usd")),
        menuItem("Similarity", tabName = "sim", icon = icon("search")),
        id = "sidebarTabs"
    )
)

body <- dashboardBody(
    tags$head(tags$style(
        ".small-box {height:120px}
     .dataTables_filter input {width:200px}"
    )),
    tabItems(
        # Overview ------------------------------------------------------------
        tabItem(
            tabName = "overview",
            fluidRow(
                valueBoxOutput("vb_avgPolar"),
                valueBoxOutput("vb_totalFunding"),
                valueBoxOutput("vb_avgSuccess")
            ),
            fluidRow(
                box(width = 6, title = "Topic KPIs", DTOutput("tbl_topicSummary")),
                box(width = 6, title = "Top Influential Actors (Overall)", DTOutput("tbl_topActorsOverall"))
            )
        ),
        # Topic Trends
        tabItem(
            tabName = "topic_tr",
            fluidRow(
                box(width = 3, selectInput("td_topic", "Select Topic", choices = sort(unique(topic_summary$topic)))),
                valueBoxOutput("td_nBills"),
                valueBoxOutput("td_funding"),
                valueBoxOutput("td_success"),
                valueBoxOutput("td_topActor")
            ),
            fluidRow(
                box(width = 6, title = "Top Actors in Topic", DTOutput("td_topActors")),
                box(width = 6, title = "Success vs Polarisation", plotlyOutput("td_scatter"))
            )
        ),
        # Actor Explorer
        tabItem(
            tabName = "actors",
            fluidRow(box(width = 3, selectInput("act_topic", "Filter by Topic", choices = c("All", sort(unique(topic_summary$topic)))))),
            tabsetPanel(
                tabPanel("Legislators", DTOutput("act_leg")),
                tabPanel("Committees", DTOutput("act_comm")),
                tabPanel("Donors", DTOutput("act_don")),
                tabPanel("Lobby Firms", DTOutput("act_lob"))
            )
        ),
        # Risk & Polarisation ------------------------------------------------
        tabItem(
            tabName = "risk",
            fluidRow(
                box(width = 12, plotlyOutput("plot_risk", height = 600))
            )
        ),
        # Funding Map ---------------------------------------------------------
        tabItem(
            tabName = "fund",
            fluidRow(box(width = 3, selectInput("fund_topic", "Select Topic", choices = sort(unique(topic_summary$topic))))),
            fluidRow(box(width = 12, sankeyNetworkOutput("sankey", height = "600px")))
        ),
        # Similarity --------------------------
        tabItem(
            tabName = "sim",
            fluidRow(
                box(
                    width = 4,
                    numericInput("sim_id", "Bill ID", value = 1, min = 1),
                    actionButton("btn_sim", "Find Similar")
                ),
                box(width = 8, DT::DTOutput("dt_sim"))
            )
        )
    )
)

ui <- dashboardPage(header, sidebar, body, skin = "blue")


# ======================================================
# SERVER
# ======================================================
server <- function(input, output, session) {
    output$vb_avgPolar <- renderValueBox({
        valueBox(round(mean(bill$polarisation_score, na.rm = TRUE), 3),
            "Avg Polarisation",
            icon = icon("adjust"), color = "yellow"
        )
    })
    output$vb_totalFunding <- renderValueBox({
        tf <- scales::dollar(sum(topic_summary$total_funding, na.rm = TRUE))
        valueBox(tf, "Total Estimated Funding", icon("dollar-sign"), color = "green")
    })
    output$vb_avgSuccess <- renderValueBox({
        av <- round(mean(topic_summary$avg_success, na.rm = TRUE), 3)
        valueBox(av, "Mean Topic Success Risk", icon("chart-line"), color = "orange")
    })

    output$tbl_topicSummary <- renderDT({
        datatable(topic_summary, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
    })

    output$tbl_topActorsOverall <- renderDT({
        datatable(actors[1:20, .(actor, type, influence, top_topic_label, bipartisan_score)],
            options = list(dom = "t"), rownames = FALSE
        )
    })

    # ─ Topic Trends ────────────────────────────────────────────────────────
    topicSel <- reactive(input$td_topic)
    topicBills <- reactive(bill[topic == topicSel()])
    topicActors <- reactive(actors[top_topic_label == topicSel()])

    output$td_nBills <- renderValueBox({
        valueBox(nrow(topicBills()), "Bills", icon = icon("file-alt"), color = "purple")
    })
    output$td_funding <- renderValueBox({
        valueBox(dollar(sum(topicBills()$lobbying_amt_sum, na.rm = TRUE)),
            "Funding",
            icon = icon("hand-holding-usd"), color = "green"
        )
    })
    output$td_success <- renderValueBox({
        valueBox(round(mean(topicBills()$success_risk, na.rm = TRUE), 3),
            "Avg Success Risk",
            icon = icon("thumbs-up"), color = "yellow"
        )
    })
    output$td_topActor <- renderValueBox({
        topA <- topicActors()[1]
        valueBox(topA$actor, "Most Influential", icon = icon("user-tie"), color = "aqua")
    })

    output$td_scatter <- renderPlotly({
        plot_ly(topicBills(),
            x = ~alignment_support, y = ~polarisation_score,
            size = ~success_risk, text = ~ paste("Bill", bill_id),
            type = "scatter", mode = "markers", sizes = c(10, 60)
        ) |>
            layout(
                xaxis = list(title = "Alignment Support"),
                yaxis = list(title = "Polarisation")
            )
    })

    # ─ Actor Explorer ──────────────────────────────────────────────────────
    actorFilter <- reactive(if (input$act_topic == "All") NULL else input$act_topic)
    filter_by_topic <- function(dt) {
        if (is.null(actorFilter())) dt else dt[topic == actorFilter()]
    }
    output$act_leg <- renderDT(datatable(filter_by_topic(leg), options = list(scrollX = TRUE)))
    output$act_comm <- renderDT(datatable(filter_by_topic(committee), options = list(scrollX = TRUE)))
    output$act_don <- renderDT(datatable(filter_by_topic(donor), options = list(scrollX = TRUE)))
    output$act_lob <- renderDT(datatable(filter_by_topic(lobby), options = list(scrollX = TRUE)))
    output$tbl_topicSummary <- renderDT({
        datatable(topic_summary, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
    })

    # ─ Risk vs Polarisation ───────────────────────────────────────────────
    output$plot_risk <- renderPlotly({
        plot_ly(topic_summary,
            x = ~avg_success, y = ~avg_polar, size = ~n_bills,
            text = ~topic, type = "scatter", mode = "markers", sizes = c(15, 70)
        ) %>%
            layout(
                xaxis = list(title = "Average Success Risk"),
                yaxis = list(title = "Average Polarisation")
            )
    })

    # ─ Funding Map Sankey ─────────────────────────────────────────────────
    output$sankey <- renderSankeyNetwork({
        dt <- donor[top_topic_label == input$fund_topic][order(-influence)][1:200]
        if (nrow(dt) == 0) {
            return(NULL)
        }
        nodes <- unique(c(dt$donor_name, dt$bill_id)) |> data.frame(name = _)
        dt[, source := match(donor_name, nodes$name) - 1]
        dt[, target := match(bill_id, nodes$name) - 1]
        sankeyNetwork(
            Links = dt, Nodes = nodes, Source = "source", Target = "target",
            Value = "influence", NodeID = "name", fontSize = 12, nodeWidth = 20
        )
    })

    # ─ Similarity Sandbox ────────────────────────────────────────────────
    output$dt_sim <- renderDT(NULL)
    observeEvent(input$btn_sim, {
        ids <- nearest(as.integer(input$sim_id), k = 10)
        df <- bill[bill_id %in% ids]
        output$dt_sim <- renderDT(datatable(df, options = list(scrollX = TRUE)))
    })
}

shinyApp(ui = ui, server = server)
