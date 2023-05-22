library(shiny)
library(shinythemes)
library(ggplot2)
library(caret)
library(pROC)
library(ROCR)
library(reshape2)
library(lmtest)

df <- read.csv("LOGIT.csv", sep = ";")

# TRATAMIENTO DE DATOS----------------------------------------------------------
# Eliminar columnas con valores iguales o muchos valores nulos
variables_eliminadas <- c("v1", "v6", "v10", "v12", "v14", "v15", "v24", "v25", "v29", "v30", "v31", "v33", "v34")
df <- df[, !(names(df) %in% variables_eliminadas)]

# Reemplazar valores texto a numerico
df$v20 <- ifelse(df$v20 == "S", 1, 0)

# Reemplazar valores faltantes con 0
df[df == ""] <- NA
df[is.na(df)] <- 0

# Obtener las columnas desde la tercera columna hasta la última
columnas <- names(df)[5:ncol(df)-1]

# Bucle para reemplazar ',' por '.'
for (col in columnas) {
  df[[col]] <- gsub(",", ".", df[[col]])
}

# Bucle para convertir las columnas a formato numérico
for (col in columnas) {
  df[[col]] <- as.numeric(df[[col]])
}

# Preprocesamiento de datos
variables_continuas <- names(df)[5:(ncol(df) - 1)]
df_variables <- df[, variables_continuas]


ui <- shinyUI(
  fluidPage(
    tags$head(
      tags$style(
        HTML("
          body {
            background-color: pink;
          }
        ")
      )
    ),
    titlePanel("Mini Desarrollo Modelo Logit con Shiny"),
    fluidRow(
      column(
        12,
        align = "center",
        h2("Bienvenidos a nuestra presentación de Mini Desarrollo Modelo Logit con Shiny!")
      )
    ),
    
    uiOutput("main_ui"),
    fluidPage(
      titlePanel("LOGIT"),
      fluidRow(
        column(
          2,
          selectInput(
            "dep_var",
            "Dependent Variable",
            choices = "target",
            selected = "target"
          )
        ),
        column(
          2,
          selectInput(
            "ind_vars",
            "Independent Variables",
            choices = setdiff(columnas, "target"),
            multiple = TRUE,
            selected = "v2"
          )
        ),
        column(
          2,
          selectInput(
            "lr_var",
            "LR-Test (Select Vars to Exclude)",
            choices = setdiff(columnas, "target"),
            multiple = TRUE,
            selected = "v2"
          )
        ),
        column(
          2,
          selectInput(
            "plot_var",
            "Plotting Variable",
            choices = columnas,
            selected = "v2"
          )
        ),
        column(
          2,
          selectInput(
            "facet_var",
            "Facet Variable",
            choices = c("target", columnas),
            selected = "target"
          )
        )
      ),
      fluidRow(
        column(
          12,
          plotOutput("plot1")
        )
      ),
      fluidRow(
        column(
          6,
          verbatimTextOutput("results")
        ),
        column(
          6,
          verbatimTextOutput("lr_test")
        )
      ),
      fluidRow(
        column(
          12,
          plotOutput("correlation")
        )
      ),
      fluidRow(
        column(
          12,
          plotOutput("roc")
        )
      )
    )
  )
)
server <- function(input, output, session) {
  # Reactive value for storing the model
  model <- reactive({
    dep_var <- input$dep_var
    ind_vars <- input$ind_vars
    if (length(ind_vars) > 0) {
      form <- as.formula(paste(dep_var, "~", paste(ind_vars, collapse = "+")))
      glm(form, family = binomial(link = "logit"), data = df)
    }
  })
  
  output$results <- renderPrint({
    model_summary <- summary(model())
    if (!is.null(model_summary))
      model_summary
  })
  
  output$lr_test <- renderPrint({
    dep_var <- input$dep_var
    ind_vars <- input$ind_vars
    lr_var <- input$lr_var
    if (is.null(lr_var)) {
      lr_var <- character(0)
    }
    lr_var <- setdiff(ind_vars, lr_var)
    if (length(lr_var) > 0) {
      form <- as.formula(paste(dep_var, "~", paste(lr_var, collapse = "+")))
      model_reduced <- glm(form, family = binomial(link = "logit"), data = df)
      lrtest(model_reduced, model())
    }
  })
  
  output$plot1 <- renderPlot({
    plot_var <- input$plot_var
    facet_var <- input$facet_var
    ggplot(df, aes(x = .data[[plot_var]])) +
      geom_density(aes(color = target), alpha = 0.6) +
      facet_wrap(~ .data[[facet_var]]) +
      theme_bw() +
      theme(legend.position = "top")
  })
  
  output$correlation <- renderPlot({
    ind_vars <- input$ind_vars
    if (!is.null(ind_vars) && length(ind_vars) > 0) {
      corr_matrix <- cor(df[, ind_vars])
      ggplot(melt(corr_matrix), aes(x = Var2, y = Var1, fill = value)) +
        geom_tile() +
        scale_fill_gradient(low = "white", high = "steelblue") +
        theme_bw()
    }
  })
  
  output$roc <- renderPlot({
    # ROC Curve
    model_roc <- roc(df$target, predict(model(), type = "response"))
    plot(model_roc, print.thres = "best", print.thres.best.method = "closest.topleft")
  })
}
  
  
shinyApp(ui, server)
