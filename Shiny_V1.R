library(shiny)
library(shinydashboard)
library(pROC)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(MASS)
library(pROC)
library(ROSE)
library(ROCR)
library(reshape2)

# CARGAR DATASET PREVIAMENTE PREPROCESADO---------------------------------------
url <- "C:/Users/Santo/Desktop/practica_modulo/LOGIT.csv"
df <- read.csv(url, sep = ";")

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

#-------------------------------------------------------------------------------
# Regresión logística
#-------------------------------------------------------------------------------
# Cargar paquetes necesarios
library(caret)

# Establecer semilla para reproducibilidad
set.seed(123)

# Crear una columna con los valores de la variable objetivo (target)
y <- df$target

# Variables independientes
variables <- colnames(df)[5:10]
X <- df[, variables]

# Crear conjunto de entrenamiento y prueba
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
train_X <- X[trainIndex, ]
train_y <- y[trainIndex]
test_X <- X[-trainIndex, ]
test_y <- y[-trainIndex]

# Realizar rebalanceo de la muestra
# Convertir la variable de respuesta a factor
train_y <- as.factor(train_y)

# Realizar rebalanceo de la muestra
train_balanced <- upSample(x = train_X, y = train_y)

# Ajustar el modelo de regresión logística
model <- glm(train_balanced$Class ~ ., data = train_balanced, family = "binomial")

# Realizar predicciones en el conjunto de testeo
probabilities <- predict(model, newdata = test_X, type = "response")

# Crear el objeto de predicción con pROC
pred <- prediction(probabilities, test_y)

# Convertir las probabilidades en clases predichas (0 o 1)
predicted_classes <- ifelse(probabilities > 0.5, 1, 0)

# Evaluar el rendimiento del modelo en el conjunto de prueba
confusion_matrix <- table(predicted_classes, test_y)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

ui <- dashboardPage(
  dashboardHeader(
    tags$li(class = "dropdown",
            tags$style(".main-header .logo {padding-top: 0px;}"),
            tags$style(".main-header .logo-text {display: none;}"),
            tags$style(".main-header .navbar {margin-left: 0px;}"),
            tags$style(".main-header .navbar-brand {color: red;}")  # Cambio de color del título
    )
  ),
  dashboardSidebar(
    sliderInput("slider", "Tolerancia:", min = 1, max = 100, value = 50),
    checkboxInput("standardize", "Estandarizar dataset", value = FALSE),
    actionButton("button1", "Graficar matriz de correlación"),
    actionButton("button2", "Graficar curva ROC")
  ),
  dashboardBody(
    navbarPage(
      tabPanel("",  # TabPanel vacío que contiene el texto "APLICACION MODELO LOGISTICO"
               tags$style(".navbar .navbar-brand {color: black;}"),  # Cambio de color del texto "APLICACION MODELO LOGISTICO"
               "APLICACION MODELO LOGISTICO"
      ),
      tabPanel("",
               textOutput("sliderValue"),
               textOutput("buttonClick"),
               plotOutput("graph")
      )
    )
  )
)

server <- function(input, output) {
  # Variable de estado para controlar qué gráfico mostrar
  state <- reactiveVal("correlation")
  
  # Función para preprocesar los datos según el estado del checkbox
  preprocessed_data <- reactive({
    if (input$standardize) {
      # Escalar los datos
      scaled_data <- scale(df_variables)
      scaled_data
    } else {
      df_variables
    }
  })
  
  output$sliderValue <- renderText({
    paste("Tolerancia actual del modelo:", input$slider)
  })
  
  observeEvent(input$button1, {
    # Cambiar el estado a "correlation" cuando se presiona el botón 1
    state("correlation")
  })
  
  observeEvent(input$button2, {
    # Cambiar el estado a "roc" cuando se presiona el botón 2
    state("roc")
  })
  
  output$graph <- renderPlot({
    data <- preprocessed_data()
    
    if (state() == "correlation") {
      # Calcular la matriz de correlación
      matriz_cor <- cor(data)
      
      # Convertir la matriz en un dataframe
      df_cor <- melt(matriz_cor)
      
      # Crear el gráfico de correlación sin el árbol
      ggplot(df_cor, aes(x = Var2, y = Var1, fill = value)) +
        geom_tile() +
        scale_fill_gradient(low = "white", high = "steelblue") +
        labs(x = NULL, y = NULL, title = "Matriz de correlación") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    } else if (state() == "roc") {
      # Realizar predicciones en el conjunto de testeo utilizando train_balanced
      # Crear el objeto de curva ROC
      roc_obj <- roc(test_y, probabilities)
      
      # Graficar la curva ROC
      plot(roc_obj, main = "Curva ROC", xlab = "Tasa de Falsos Positivos", ylab = "Tasa de Verdaderos Positivos")
      lines(x = c(0, 1), y = c(0, 1), lty = 2, col = "gray")  # Línea de referencia
      legend("bottomright", legend = c("Curva ROC"), lty = 1, col = "black")
    }
  })
}

shinyApp(ui, server)
