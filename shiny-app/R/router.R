# Routing helpers -----------------------------------------------------------------------

parse_query <- function(session) {
  shiny::parseQueryString(isolate(session$clientData$url_search))
}

encode_query <- function(params) {
  if (!length(params)) return("")
  pairs <- purrr::imap_chr(params, function(value, key) {
    paste0(utils::URLencode(key, reserved = TRUE), "=", utils::URLencode(as.character(value), reserved = TRUE))
  })
  paste0("?", paste(pairs, collapse = "&"))
}

update_query <- function(session, params, mode = c("replace", "push")) {
  mode <- match.arg(mode)
  current <- parse_query(session)
  merged <- modifyList(current, params)
  merged <- merged[!vapply(merged, is.null, logical(1))]
  shiny::updateQueryString(encode_query(merged), mode = mode, session = session)
}

get_query_value <- function(session, key, default = NULL) {
  query <- parse_query(session)
  if (!is.null(query[[key]])) query[[key]] else default
}
