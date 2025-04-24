##mae method to be used in SuperLearner
method.mae <- function() {
  out <- list(
   # require = 'nnls',
    computeCoef = function(Z, Y, libraryNames, verbose, obsWeights, ...) {
      # compute cvRisk
      cvRisk <- apply(Z, 2, function(x) mean(obsWeights * abs(x - Y)))
      names(cvRisk) <- libraryNames
      
      mean_absolute_loss <- function(actual, predicted) {
        mean(abs(actual - predicted))
      }
      
      # Define the objective function for optimization
      objective_function <- function(beta, x, y) {
        predicted_y <- beta %*% t(x)  
        mean_absolute_loss(y, predicted_y)
      }
      
      minimize_mae <- function(Z, Y) {
        # Initial guess for coefficients (beta_0, beta_1)
        initial_guess <- rep(0, dim(Z)[2])
        
        # Use optimization function (e.g., optim) to minimize the objective function
        result <- optim(initial_guess, objective_function, 
                        x = Z, y = Y, method = "L-BFGS-B",
                        control = list(maxit = 2000))
        
        # Extract optimized coefficients
        result
      }
      
      fit.mae <- minimize_mae(Z, Y)
      
      if (verbose) {
        message(paste("MAE convergence:", fit.mae$convergence == 0))
      }
      
      initCoef <- fit.mae$par
      initCoef[is.na(initCoef)] <- 0.0
      # normalize so sum(coef) = 1 if possible
      if (sum(initCoef) > 0) {
        coef <- initCoef / sum(initCoef)
      } else {
        warning("All algorithms have zero weight", call. = FALSE)
        coef <- initCoef
      }
      out <- list(cvRisk = cvRisk, coef = coef, optimizer = fit.mae)
      return(out)
    },
    
    computePred = function(predY, coef, ...) {
      if (sum(coef != 0) == 0) {
        warning("All metalearner coefficients are zero, predictions will all be equal to 0", call. = FALSE)
        out <- rep(0, nrow(predY))
      } else {
        # Restrict crossproduct to learners with non-zero coefficients.
        out <- crossprod(t(predY[, coef != 0, drop = FALSE]), coef[coef != 0])
      }
      return(out)
    }
  )
  invisible(out)
}



