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

fmt_pct1 <- function(x) {
    res <- scales::percent(x, accuracy = 0.1)
    res[is.na(x)] <- "—"
    res
}

fmt_date <- function(x) {
    if (inherits(x, "Date")) {
        format(x, "%b %d, %Y")
    } else {
        x
    }
}

pal_sequential <- function(n = 7L) {
    grDevices::colorRampPalette(c("#e8f0fe", "#2B4C7E"))(n)
}

pal_diverging <- function(n = 7L) {
    grDevices::colorRampPalette(c("#81312f", "#f5f2f0", "#0F766E"))(n)
}

rank_and_percentile <- function(x, decreasing = TRUE) {
    if (!length(x)) {
        return(list(rank = integer(), percentile = numeric()))
    }
    ord <- if (decreasing) -x else x
    ranks <- rank(ord, ties.method = "min")
    pct <- ranks / max(ranks)
    list(rank = ranks, percentile = pct)
}

safe_pull <- function(data, column, default = NA) {
    if (column %in% names(data)) {
        data[[column]]
    } else {
        default
    }
}

sanitize_id <- function(x) {
    stringr::str_replace_all(tolower(x), "[^a-z0-9]+", "-")
}

with_cache <- function(expr, cache_key) {
    shiny::bindCache(expr, cache_key, cache = cachem::cache_disk(dir = tempfile("shiny-cache-")))
}
