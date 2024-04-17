# rm(list = ls())
# library(randomForest)
# library(permimp) # this is for CPI
# library(sensitivity)
# library(pracma)

data.generator <- function(n, p, Sigma) {
  
  
  P <- matrix(Sigma, nrow = p)
  # Gaussian copula
  X <- pnorm(matrix(rnorm(n * p), ncol = p) %*% chol(P))
  X <- X - 0.5
  y <- 5 * X[, 1] + 15 * X[, 5] + 20 * X[, 3]^2 + 2 * X[, 9] +
    10 * sin(pi * (X[, 9] - 0.5) * (X[, 10] - 0.5)) + rnorm(n)
  
  return(list('X' = X, 'y' = y))
}




n <- 500 # 500
p <- 100 # 100
rho <- 0.8 # 0.8


Sigma <- array(0, c(p, p))
for (k in 1:p) {
  for (l in 1:p) {
    Sigma[k, l] <- rho^(abs(k-l))
  }
}


# Run the experiments 100 times for MDI, MDA, and CPI
R <- 100
MDI <- array(0, c(100, 100))
CPI <- array(0, c(100, 100))
for (r in 1:R) {
  sample.data <- data.generator(n, p, Sigma)
  
  x.vector <- sample.data[['X']]
  y <- sample.data[['y']]
  V <- as.data.frame(cbind(y, x.vector))
  
  # hyperparameter tuning : (40, 5, 100) for labmda = 0.8
  # hyperparameter tuning : (40, 5, 90) for labmda = 0.1
  
  rf.obj <- randomForest(y ~ ., data = V, maxnodes = 40, nodesize = 5, mtry = 90, importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE)
  MDI[, r] <- rf.obj$importance[, 2]
  # imp.cpi <- permimp(rf.obj, conditional = TRUE, do_check = FALSE, progressBar = TRUE) # default threshold = 0.95
  # CPI[, r] <- imp.cpi$values
  print(paste0('round: ', r))
}

significant_list <- c(1, 3, 5, 9, 10)
min(CPI[, 1][significant_list])
max(CPI[, 1][-significant_list])

counter_ <- 0
for (r in 1:R) {
  if (min(MDI[, r][significant_list]) <= max(MDI[, r][-significant_list])) {
    counter_ <- counter_ + 1
  }
}
counter_




# Save an object to a file
object = list('MDI' = MDI, 'CPI' = CPI)
# file = '/Users/xbb/Dropbox/Collaborative-Trees-Ensemble/simulated_data/example_bias_mdi_cpi_high.rds'
file = '/Users/xbb/Dropbox/Collaborative-Trees-Ensemble/simulated_data/example_bias_mdi_cpi_low.rds'
# saveRDS(object, file = file)
# Restore the object
# object <- readRDS(file = file)


counter_MDI <- 0
counter_CPI <- 0
for (r in 1:R) {
  if (min(object['MDI'][[1]][, r][significant_list]) <= 
      max(object['MDI'][[1]][, r][-significant_list])) {
    counter_MDI <- counter_MDI + 1
  }
  
  if (min(object['CPI'][[1]][, r][significant_list]) <= 
      max(object['CPI'][[1]][, r][-significant_list])) {
    counter_CPI <- counter_CPI + 1
  }
  
}

counter_MDI
counter_CPI

## Tuning CPI and MDI



maxnodes_ <- c(10, 20, 30, 40)
nodesize_ <- c(5, 10, 15, 20)
mtry_ <- c(floor(p * 0.5), floor(p * 0.7), floor(p * 0.9), p)
hyper_ <- array(0, c(4 * 4 * 4, 3))
index <- 1
for (a1 in maxnodes_) {
  for (a2 in nodesize_) {
    for (a3 in mtry_) {
      hyper_[index, ] <- c(a1, a2, a3)
      index <- index + 1
    }
  }
}
# hyperparameter tuning : (10 15 27) for labmda = 0.8
# hyperparameter tuning : (?, ?, ?) for labmda = 0.1


Sigma <- array(0, c(p, p))
for (k in 1:p) {
  for (l in 1:p) {
    Sigma[k, l] <- rho^(abs(k-l))
  }
}


significant_list <- c(1, 3, 5, 9, 10)

for (run_ in 1:dim(hyper_)[1]) {
  negative.count.mdi <- 0
  for (r in 1:30) {
    sample.data <- data.generator(n, p, Sigma)
    
    x.vector <- sample.data[['X']]
    y <- sample.data[['y']]
    X <- as.data.frame(cbind(x.vector, y))
    
    maxnodes <- hyper_[run_, 1]
    nodesize <- hyper_[run_, 2]
    mtry <- hyper_[run_, 3]
    rf.obj <- randomForest(y ~ ., data = X, maxnodes = maxnodes, nodesize = nodesize, mtry = mtry,
                           keep.forest = TRUE, keep.inbag = TRUE, importance = TRUE)
    
    if (max(rf.obj$importance[, 2][-significant_list] )
        >= min(rf.obj$importance[, 2][significant_list])) {
      negative.count.mdi <- negative.count.mdi + 1
    }

  }
  # print(paste0("Rates of wrongs, CPI: ", negative.count.cpi))
  # print(paste0("Rates of wrongs, MDI: ", negative.count.mdi))
  print(hyper_[run_, ])
  print(paste0(' : ', negative.count.mdi))
}



##
##
###
###
# Sobol indices

n <- 500 # 500
p <- 100 # 100
rho <- 0.1 # 0.8



Sigma <- array(0, c(p, p))
for (k in 1:p) {
  for (l in 1:p) {
    Sigma[k, l] <- rho^(abs(k-l))
  }
}


maxnodes_ <- c(10, 20, 30, 40)
nodesize_ <- c(5, 10, 15, 20)
mtry_ <- c(60, 70, 80, 90)
hyper_ <- array(0, c(4 * 4 * 4, 3))
index <- 1
for (a1 in maxnodes_) {
  for (a2 in nodesize_) {
    for (a3 in mtry_) {
      hyper_[index, ] <- c(a1, a2, a3)
      index <- index + 1
    }
  }
}
# hyperparameter tuning : (40, 15, 90) for labmda = 0.8
# hyperparameter tuning : (?, ?, ?) for labmda = 0.1

counter_record <- array(0, c(dim(hyper_)))

for (run_ in 1:dim(hyper_)[1]) {
  R <- 100
  sobol.v <- list()
    
  
  
  for (r in 1:R) {
    sample.data <- data.generator(n, p, Sigma)
    
    x.vector <- sample.data[['X']]
    y <- sample.data[['y']]
    X <- as.data.frame(cbind(x.vector, y))
    
    maxnodes <- hyper_[run_, 1]
    nodesize <- hyper_[run_, 2]
    mtry <- hyper_[run_, 3]
    rf.obj <- randomForest(y ~ ., data = X, maxnodes = maxnodes, nodesize = nodesize, mtry = mtry,
                           keep.forest = TRUE, keep.inbag = TRUE)
    
    
    n_ <- dim(x.vector)[1]
    X1 <- x.vector[(1:floor(n_ / 2)),]
    X2 <- x.vector[((floor(n_ / 2) + 1): n_),]
    V1 <- data.frame(X1)
    V2 <- data.frame(X2)
    colnames(V1) <- colnames(X)[1:10]
    colnames(V2) <- colnames(X)[1:10]
  
    results <- sobolSalt(model = rf.obj, X1, X2, scheme="B", nboot = 100)
    
    
    # total effect
    # results$T[, 1]
    # first order
    # results$S[, 1]
    # second order
    # results$S2[, 1]
    results$S2[, 1][length(results$S2[, 1])]
  
    sobol.v[[r]] <- list()
    sobol.v[[r]][[1]] <- results$T[, 1]
    sobol.v[[r]][[2]] <- results$S[, 1]
    sobol.v[[r]][[3]] <- results$S2[, 1]
    
    
    
    # 100 + 98 + 97 + 96 + 95 + 94 + 93 + 92
    # rownames(results$S2)[c(1:9, 100:107, 198:204, 295:300, 391:395, 486:489, 580:582, 673:674, 765)]
    # results$S2[c(1:9, 100:107, 198:204, 295:300, 391:395, 486:489, 580:582, 673:674, 765), ]
    
    # sobol.v[r, ] <- c(results$T[, 1][1:10], results$S[, 1][1:10], results$S2[, 1][c(1:9, 100:107, 198:204, 295:300, 391:395, 486:489, 580:582, 673:674, 765)])
    # sobol.v[r, ] <- c(results$T[, 1][1:10], results$S[, 1][1:10], results$S2[, 1])
    
    # print(paste0('round: ', r))
  }
  sobol.v
  # sobol.v.temp.highbias <- sobol.v
  
  counter <- c(0, 0, 0)
  for (r in 1:R) {
    indice_list <- sobol.v[[r]]
    
    total.v <- indice_list[[1]]
    first.v <- indice_list[[2]]
    second.v <- indice_list[[3]]
    
    # total effect
    if (min(total.v[c(1, 3, 5, 9, 10)]) >= max(total.v[-c(1, 3, 5, 9, 10)])) {
      counter[1] <- counter[1] + 1
    }
    # first order
    if(min(first.v[c(1, 3, 5)]) >= max(first.v[-c(1, 3, 5, 9, 10)])) {
      counter[2] <- counter[2] + 1
    }
    # second order
    if (second.v[765] >= max(second.v[-765])) {
      counter[3] <- counter[3] + 1
    }
    
  }
  counter_record[run_, ] <- counter
  print(counter_record)
  print(hyper_[run_, ])
}



