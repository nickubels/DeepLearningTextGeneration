# Loading data
all_content <- readLines("~/Dropbox/University/Year\ 4/Deep\ Learning/results.csv")
skip_second <- all_content[-2]
data <- read.csv(textConnection(skip_second), header = TRUE, stringsAsFactors = FALSE)

# Removing incomplete results
data <- data[data$Finished == 1,]
