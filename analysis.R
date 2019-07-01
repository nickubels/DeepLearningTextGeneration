# Loading data
all_content <- readLines("~/Dropbox/University/Year\ 4/Deep\ Learning/results.csv")
skip_second <- all_content[-2]
data <- read.csv(textConnection(skip_second), header = TRUE, stringsAsFactors = FALSE)

# Removing incomplete results
data <- data[data$Finished == 1,]

# Summary score
summary(data$SC0)
sd(data$SC0)

# Get counts
generated_correct <- 0
generated_incorrect <- 0
generated_incorrect <- generated_incorrect + sum(data$Q1 == 1)
generated_correct <- generated_correct + sum(data$Q1 == 2)
generated_incorrect <- generated_incorrect + sum(data$Q2 == 1)
generated_correct <- generated_correct + sum(data$Q2 == 2)
generated_incorrect <- generated_incorrect + sum(data$Q3 == 1)
generated_correct <- generated_correct + sum(data$Q3 == 2)
generated_incorrect <- generated_incorrect + sum(data$Q4 == 1)
generated_correct <- generated_correct + sum(data$Q4 == 2)
generated_incorrect <- generated_incorrect + sum(data$Q5 == 1)
generated_correct <- generated_correct + sum(data$Q5 == 2)
generated_incorrect <- generated_incorrect + sum(data$Q6 == 1)
generated_correct <- generated_correct + sum(data$Q6 == 2)
generated_incorrect <- generated_incorrect + sum(data$Q7 == 1)
generated_correct <- generated_correct + sum(data$Q7 == 2)
generated_incorrect <- generated_incorrect + sum(data$Q8 == 1)
generated_correct <- generated_correct + sum(data$Q8 == 2)

real_correct <- 0
real_incorrect <- 0
real_incorrect <- real_incorrect + sum(data$Q9 == 2)
real_correct <- real_correct + sum(data$Q9 == 1)
real_incorrect <- real_incorrect + sum(data$Q10 == 2)
real_correct <- real_correct + sum(data$Q10 == 1)
real_incorrect <- real_incorrect + sum(data$Q11 == 2)
real_correct <- real_correct + sum(data$Q11 == 1)
real_incorrect <- real_incorrect + sum(data$Q12 == 2)
real_correct <- real_correct + sum(data$Q12 == 1)

# Make table
trump <- c(real_correct, real_incorrect)
generated <- c(generated_incorrect, generated_correct)
tbl <- matrix(c(real_correct, real_incorrect, generated_incorrect, generated_correct),nrow=2,byrow=TRUE)

# chisq
chisq.test(tbl)
