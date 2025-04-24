##huber method to be used in SuperLearner
##Another example from the link
## https://github.com/wuziyueemory/Huber-loss-based-SuperLearner/blob/main/HuberSL.R
method.huber <- function() {
  out <- list(
    # require = 'nnls',
    computeCoef = function(Z, Y, libraryNames, verbose, obsWeights, delta = 0.1, 
                           ...) {
      # compute cvRisk
      huber_loss <- function(actual, predicted, delta) {
        residuals <- actual - predicted
        loss <- ifelse(abs(residuals) <= delta, 0.5 * residuals^2, 
                       delta * (abs(residuals) - 0.5 * delta))
        return(mean(loss))
      }
      cvRisk <- apply(Z, 2, function(x) huber_loss(Y,x,delta)) 
      names(cvRisk) <- libraryNames
      
      # Define the objective function for optimization
      objective_function_huber <- function(bb, x, y, delta) {
        predicted_y <- bb %*% t(x)  
        huber_loss(y, predicted_y, delta)
      }
      
      minimize_huber <- function(Z, Y, delta) {
        # Initial guess for coefficients (beta_0, beta_1)
        initial_guess <- rep(0, dim(Z)[2])
        
        # Use optimization function (e.g., optim) to minimize the objective function
        result <- optim(initial_guess, objective_function_huber, 
                        x = Z, y = Y, method = "L-BFGS-B",
                        delta = delta)
        # Extract optimized coefficients
        result
      }
      
      fit.huber <- minimize_huber(Z, Y,delta)
      
      if (verbose) {
        message(paste("Huber loss convergence:", fit.huber$convergence == 0)) ##0 indicated successful completion
      }
      
      initCoef <- fit.huber$par
      initCoef[is.na(initCoef)] <- 0.0
      # normalize so sum(coef) = 1 if possible
      if (sum(initCoef) > 0) {
        coef <- initCoef / sum(initCoef)
      } else {
        warning("All algorithms have zero weight", call. = FALSE)
        coef <- initCoef
      }
      out <- list(cvRisk = cvRisk, coef = coef, optimizer = fit.huber)
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



