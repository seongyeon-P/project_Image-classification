use_python("C:/Users/tjddu/AppData/Local/Programs/Python/Python39/python.exe")
library(reticulate)

keras <- import("tensorflow.keras")
np <- import("numpy")

cifar10_labels <- c("airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck")

model <- keras$models$load_model("cnn_model.h5")

predict_cnn <- function(input_array) {
  input_array <- array_reshape(input_array, c(1, 32, 32, 3))
  preds <- model$predict(input_array)
  
  class_id <- np$argmax(preds)
  confidence <- round(np$max(preds) * 100, 2)
  list(class_id = class_id, confidence = confidence)
}

library(shiny)
library(magick)

ui <- fluidPage(
  titlePanel("CNN 이미지 분류"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("image", "이미지 업로드(JPG/PNG)", accept = c(".png", ".jpg", ".jpeg")),
      actionButton("predict", "예측 실행")
    ),
    
    mainPanel(
      h4("업로드된 이미지"),
      imageOutput("uploaded_img", height = 200),
      hr(),
      h4("예측 결과"),
      verbatimTextOutput("result")
    )
  )
)

server <- function(input, output, session) {
  
  # 이미지 표시 (observe 밖으로 이동)
  output$uploaded_img <- renderImage({
    req(input$image)
    list(
      src = input$image$datapath,
      contentType = input$image$type,
      width = 200
    )
  }, deleteFile = FALSE)
  
  # 예측 실행 버튼 클릭 시
  observeEvent(input$predict, {
    if (is.null(input$image)) {
      output$result <- renderText({
        "이미지를 먼저 업로드해주세요."
      })
      return()
    }
    
    tryCatch({
      # 1. 이미지 전처리
      img <- image_read(input$image$datapath)
      img <- image_resize(img, "32x32!")
      
      img_array <- as.integer(img[[1]])
      img_array <- img_array / 255
      dim(img_array) <- c(32, 32, 3)
      img_array <- array(img_array, dim = c(1, 32, 32, 3))
    
      # 2. 예측
      pred <- predict_cnn(img_array)
      class_id <- pred$class_id
      confidence <- pred$confidence
      class_name <- cifar10_labels[class_id + 1]
      
      # 3. 결과 출력
      output$result <- renderText({
        paste0("예측된 클래스: ", class_id, " (", class_name, ")\n",
               "신뢰도: ", confidence, "%")
      })
      
    }, error = function(e) {
      output$result <- renderText({
        paste("오류 발생:", e$message)
      })
    })
  })}

shinyApp(ui, server)
