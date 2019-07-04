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
generated_pre_correct <- 0
generated_pre_incorrect <- 0
generated_pre_incorrect <- generated_pre_incorrect + sum(data$Q1 == 1)
generated_pre_correct <- generated_pre_correct + sum(data$Q1 == 2)
generated_pre_incorrect <- generated_pre_incorrect + sum(data$Q2 == 1)
generated_pre_correct <- generated_pre_correct + sum(data$Q2 == 2)
generated_pre_incorrect <- generated_pre_incorrect + sum(data$Q3 == 1)
generated_pre_correct <- generated_pre_correct + sum(data$Q3 == 2)
generated_pre_incorrect <- generated_pre_incorrect + sum(data$Q4 == 1)
generated_pre_correct <- generated_pre_correct + sum(data$Q4 == 2)

generated_non_correct <- 0
generated_non_incorrect <- 0
generated_non_incorrect <- generated_non_incorrect + sum(data$Q5 == 1)
generated_non_correct <- generated_non_correct + sum(data$Q5 == 2)
generated_non_incorrect <- generated_non_incorrect + sum(data$Q6 == 1)
generated_non_correct <- generated_non_correct + sum(data$Q6 == 2)
generated_non_incorrect <- generated_non_incorrect + sum(data$Q7 == 1)
generated_non_correct <- generated_non_correct + sum(data$Q7 == 2)
generated_non_incorrect <- generated_non_incorrect + sum(data$Q8 == 1)
generated_non_correct <- generated_non_correct + sum(data$Q8 == 2)

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
tbl <- matrix(c(real_correct, real_incorrect, generated_pre_incorrect+generated_non_incorrect, generated_pre_correct+generated_non_correct),nrow=2,byrow=TRUE)
tbl2 <- matrix(c(real_correct/944,
                 real_incorrect/944,
                 generated_pre_incorrect+generated_non_incorrect/1888,
                 generated_pre_correct+generated_non_correct/1888),nrow=2,byrow=TRUE)

# chisq
chisq.test(tbl)

tbl_pre <- matrix(c(real_correct, real_incorrect, generated_pre_incorrect, generated_pre_correct),nrow=2,byrow=TRUE)
tbl_non <- matrix(c(real_correct, real_incorrect, generated_non_incorrect, generated_non_correct),nrow=2,byrow=TRUE)

chisq.test(tbl_pre)
chisq.test(tbl_non)