##superlearner wrappers for quantile regression and QRNN
SL.quantreg <- function(Y, X, newX, family, obsWeights, model = TRUE, ...) {
  
  # X must be a matrix.
  if (!is.matrix(X)) {
    X = as.matrix(X)
  }
  
  fit.quantreg <- rq(Y ~ ., data = X, 
                     weights = obsWeights,
                 tau = 0.5, method = 'br')
  
  # newX must be a dataframe, not a matrix.
  if (is.data.frame(newX)) {
    newX = as.matrix(newX)
  }
  
  pred <- predict(fit.quantreg, newdata = newX)
  fit <- list(object = fit.quantreg)
  class(fit) <- "SL.quantreg"
  out <- list(pred = pred, fit = fit)
  return(out)
}


predict.SL.quantreg <- function(object, newdata, ...) {
  # newdata must be a dataframe, not a matrix.
  if (is.matrix(newdata)) {
    newdata = as.data.frame(newdata)
  }
  pred <- predict(object = object$object, newdata = newdata)
  pred
}

##### SL wrapper for quantile random forest
SL.QRforest <- function(Y, X, newX, family, obsWeights, 
                        model = TRUE, ...) {
  
  # X must be a dataframe, not a matrix.
  if (is.matrix(X)) {
    X = as.data.frame(X)
  }
  
  fit.quantreg <- quantregForest(x =X, y = Y)
  
  # newX must be a dataframe, not a matrix.
  if (is.matrix(newX)) {
    newX = as.data.frame(newX)
  }
  
  pred <- predict(fit.quantreg, newdata = newX, what= 0.5)
  fit <- list(object = fit.quantreg)
  class(fit) <- "SL.QRforest"
  out <- list(pred = pred, fit = fit)
  return(out)
}


predict.SL.QRforest  <- function(object, newdata, ...) {
  # newdata must be a dataframe, not a matrix.
  if (is.matrix(newdata)) {
    newdata = as.data.frame(newdata)
  }
  pred <- predict(object = object$object, newdata = newdata,
                  what= 0.5)
  pred
}

######## SL wrapper for quantile random forest using ranger
SL.ranger.qr <-
  function(Y, X, newX, family,
           obsWeights,
           num.trees = 500,
           mtry = floor(sqrt(ncol(X))),
           write.forest = TRUE,
        #   probability = family$family == "binomial",
        #  min.node.size = ifelse(family$family == "gaussian", 5, 1),
          # replace = TRUE,
          # sample.fraction = ifelse(replace, 1, 0.632),
          # num.threads = 1,
          verbose = T,
        quantreg = TRUE,
        keep.inbag=TRUE,
           ...) {
    # need write.forest = TRUE for predict method
    
    
    if (family$family == "binomial") {
      Y = as.factor(Y)
    }
    
    # Ranger does not seem to work with X as a matrix, so we explicitly convert to
    # data.frame rather than cbind. newX can remain as-is though.
    if (is.matrix(X)) {
      X = data.frame(X)
    }
    
    # Use _Y as our outcome variable name to avoid a possible conflict with a
    # variable in X named "Y".
    fit <- ranger::ranger(`_Y` ~ ., data = cbind("_Y" = Y, X),
                          num.trees = num.trees,
                          mtry = mtry,
                       #   min.node.size = min.node.size,
                         # replace = replace,
                          #sample.fraction = sample.fraction,
                        #  case.weights = obsWeights,
                          write.forest = write.forest,
                          #probability = probability,
                         # num.threads = num.threads,
                          verbose = verbose,
                          quantreg = quantreg,
                          keep.inbag=keep.inbag)
    
    pred <- predict(fit, data = newX, type = "quantiles", quantiles = c(0.5))$predictions
    
    fit <- list(object = fit, verbose = verbose)
    class(fit) <- c("SL.ranger.qr")
    out <- list(pred = pred, fit = fit)
    return(out)
  }





predict.SL.ranger.qr <- function(object, newdata, family,
                             num.threads = 1,
                             verbose = object$verbose,
                             type = "quantiles", quantiles = c(0.5),
                             ...) {
  
  # Binomial and gaussian prediction is the same.
  pred <- predict(object$object, data = newdata, verbose = verbose,
                  num.threads = num.threads, type = type, quantiles = quantiles)$predictions
  
  pred
}


### SL wrapper for quantile neural network using R package qrnn
SL.qrnn <- function(Y, X, newX, family, obsWeights, 
                        model = TRUE, ...) {
  
  # X must be a dataframe, not a matrix.
  if (!is.matrix(X)){
    X = as.matrix(X)
  }
  
  if (!is.matrix(Y)) {
    Y= as.matrix(Y)
  }
  
  fit.quantreg <- qrnn.fit(x=X, y=Y, n.hidden=1, tau=0.5, lower=0,
                                      iter.max=200, n.trials=3)
  
  # newX must be a dataframe, not a matrix.
  if (!is.matrix(newX)) {
    newX = as.matrix(newX)
  }
  
  pred <- qrnn.predict(x = newX, parms = fit.quantreg)
  
  fit <- list(object = fit.quantreg)
  class(fit) <- "SL.qrnn"
  out <- list(pred = pred, fit = fit)
  return(out)
}


predict.SL.qrnn  <- function(object, newdata, ...) {
  # newdata must be a dataframe, not a matrix.
  if (!is.matrix(newdata)) {
    newdata = as.matrix(newdata)
  }
  pred <- qrnn.predict(x = newX, parms = object$object)
  pred
}



