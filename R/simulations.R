


#' simulateHTS
#' 
#' This function simulates a set of (x, y) pairs.
#' The covariates x are compositional, meaning they only
#' carry relative information.
#' The response y is a binary indicator.
#' The rule linking x and y can be a balance or an amalgamation.
#'
#' @param n Number of observations
#' @param p Number of covariates
#' @param logratio A string indicating 'simple', 'balance', or 
#'     'amalgamation'
#'
#' @return A list containing a matrix of inputs and a vector of outputs
simulateHTS = function(n, p, logratio = 'simple'){
  
  # Simulate independent variables
  alpha0 = rep(1.0, p) / log(p)
  alpha = gtools::rdirichlet(1, alpha0)
  alpha = sort(alpha, decreasing=T)
  X = matrix(0.0, n, p)
  P = matrix(0.0, n, p)
  numCounts = stats::rpois(n, 10 * p)
  for (i in 1:n) {
    classProb = gtools::rdirichlet(1, alpha)
    x = stats::rmultinom(1, numCounts[i], classProb)
    # X[i,] = x / sum(x)
    X[i,] = x
    P[i,] = classProb
  }
  
  # Simulate dependent variable
  if (logratio == 'simple') {
    if (p < 2) {
      stop("Input dimension must be >= 2")
    }
    eta = log(P[, 1]) - log(P[, 2])
  } else if (logratio == 'balance') {
    if (p < 10) {
      stop("Input dimension must be >= 10")
    }
    eta = rowSums(log(P[, c(1,2,6)])) - rowSums(log(P[, c(3,8)]))
  } else if (logratio == 'amalgamation') {
    if (p < 20) {
      stop("Input dimension must be >= 20")
    }
    eta = log(rowSums(P[, c(1,2,6,7,15)])) - log(rowSums(P[, c(3,8,16,17)]))
  } else {
    stop("Variable logratio incorrectly specified.")
  }
  
  outProb = 1 / (1 + exp(-(eta - mean(eta)))) * 1.0
  y = stats::rbinom(n, 1, outProb)
  
  return(list(x=data.frame(X), y=data.frame(y)))
}
