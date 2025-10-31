#' Application theme configuration
#'
#' Defines the shared bslib theme for the dashboard, including
#' typography, color palette, and dark-mode defaults.
app_theme <- function() {
  bslib::bs_theme(
    version = 5,
    preset = "bootstrap",
    base_font = bslib::font_face(
      family = "Inter",
      file = "fonts/Inter-Variable.woff2",
      weight = 400
    ),
    heading_font = bslib::font_face("Inter", file = "fonts/Inter-Variable.woff2"),
    primary = "#2B4C7E",
    secondary = "#0F766E",
    success = "#0F766E",
    light = "#f8f9fa",
    dark = "#1f2933"
  )
}
