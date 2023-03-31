
# Here we implement the codacore model

library(keras)
utils::globalVariables(c("self"))

# """Fits a single base learner"""
# Private class not to be called by user
.CoDaBaseLearner <- function(
  x,
  y,
  boostingOffset,
  logRatioType,
  objective,
  lambda,
  cvParams,
  optParams,
  verbose
){
  
  cdbl = list(
    intercept=NULL,
    slope=NULL,
    weights=NULL,
    softAssignment=NULL,
    hard=NULL,
    x=x,
    y=y,
    boostingOffset=boostingOffset,
    logRatioType=logRatioType,
    objective=objective,
    lambda=lambda,
    cvParams=cvParams,
    optParams=optParams,
    verbose=verbose
  )
  class(cdbl) = "CoDaBaseLearner"
  
  # Train the relaxation model
  cdbl = trainRelaxation.CoDaBaseLearner(cdbl)
  
  # Find optimal cutoff by CV
  cutoff = findBestCutoff.CoDaBaseLearner(cdbl)
  
  # Use cutoff to "harden" the log-ratio
  cdbl = harden.CoDaBaseLearner(cdbl, cutoff)
  
  # And recompute the linear coefficients
  cdbl = setInterceptAndSlope.CoDaBaseLearner(cdbl, cdbl$x, cdbl$y, cdbl$boostingOffset)
  
  # Add some metrics
  yHat = predict(cdbl, x) + boostingOffset
  if (cdbl$objective == 'binary classification') {
    cdbl$ROC = pROC::roc(y, yHat, quiet=TRUE)
    cdbl$AUC = pROC::auc(cdbl$ROC)
    cdbl$accuracy = mean(y == (yHat > 0))
  } else {
    cdbl$RMSE = sqrt(mean((y - yHat)^2))
    cdbl$Rsquared = 1 - cdbl$RMSE^2 / stats::var(y)
  }
  
  return(cdbl)
}


#' @import keras
trainRelaxation.CoDaBaseLearner = function(cdbl) {
  startTime = Sys.time()
  
  # Set up traininable variables
  inputDim = ncol(cdbl$x)
  numObs = nrow(cdbl$x)
  
  # Initializaing the intercept at the average of the data
  # this helps optimization greatly
  # TODO: should experiment with slopeInit parameter for potential gains
  if (cdbl$objective == "binary classification") {
    loss_func = 'binary_crossentropy'
    if (abs(mean(1 / (1 + exp(-cdbl$boostingOffset))) - mean(cdbl$y)) < 0.001) {
      # Protect against numerical errors in glm() call
      interceptInit = 0.0
    } else {
      tempGLM = stats::glm(cdbl$y ~ 1, offset=cdbl$boostingOffset, family='binomial')
      interceptInit = tempGLM$coef[[1]]
    }
    slopeInit = 0.1
    metrics = c('accuracy')
  } else if (cdbl$objective == "regression") {
    loss_func = 'mean_squared_error'
    interceptInit = mean(cdbl$y - cdbl$boostingOffset)
    slopeInit = 0.1 # * stats::sd(cdbl$y - cdbl$boostingOffset)
    metrics = c('mean_squared_error')
  }
  
  # Define the forward pass for our relaxation,
  # which differs for balances and amalgamations
  if (cdbl$logRatioType == 'A') {
    epsilon = cdbl$optParams$epsilonA
    forwardPass = function(x, mask = NULL) {
      softAssignment = 2 * keras::k_sigmoid(self$weights) - 1
      # Add the small value to ensure gradient flows at exact zeros (initial values)
      pvePart = keras::k_dot(x, keras::k_relu(softAssignment + 1e-20))
      nvePart = keras::k_dot(x, keras::k_relu(-softAssignment))
      logRatio = keras::k_log(pvePart + epsilon) - 
        keras::k_log(nvePart + epsilon)
      eta = self$slope * logRatio + self$intercept + self$boostingOffset
      # keras::k_sigmoid(eta)
      eta
    }
  } else if (cdbl$logRatioType == 'B') {
    epsilon = cdbl$optParams$epsilonB
    forwardPass = function(x, mask = NULL) {
      softAssignment = 2 * keras::k_sigmoid(self$weights) - 1
      # Add the small value to ensure gradient flows at exact zeros (initial values)
      pvePart = keras::k_relu(softAssignment + 1e-20)
      nvePart = keras::k_relu(-softAssignment)
      logRatio = keras::k_dot(keras::k_log(x), pvePart) / keras::k_maximum(keras::k_sum(pvePart), epsilon) -
        keras::k_dot(keras::k_log(x), nvePart) / keras::k_maximum(keras::k_sum(nvePart), epsilon)
      eta = self$slope * logRatio + self$intercept + self$boostingOffset
      # keras::k_sigmoid(eta)
      eta
    }
  }
  
  if (FALSE) {
    tensorflow::tf$random$set_seed(0)
  }
  
  # Set up custom layer
  CustomLayer <- R6::R6Class(
    "CustomLayer",
    
    inherit = keras::KerasLayer,
    
    public = list(
      output_dim = NULL,
      weights = NULL,
      intercept = NULL,
      slope = NULL,
      boostingOffset = NULL,
      # epsilon = NULL,
      
      initialize = function() {
        self$output_dim <- 1
      },
      
      build = function(input_shape) {
        self$weights <- self$add_weight(
          name = 'weights', 
          shape = list(as.integer(inputDim), as.integer(1)),
          initializer = keras::initializer_zeros(),
          trainable = TRUE
        )
        self$intercept <- self$add_weight(
          name = 'intercept', 
          shape = list(as.integer(1)),
          initializer = keras::initializer_constant(interceptInit),
          trainable = TRUE
        )
        self$slope <- self$add_weight(
          name = 'slope', 
          shape = list(as.integer(1)),
          initializer = keras::initializer_constant(slopeInit),
          trainable = TRUE
        )
        self$boostingOffset <- self$add_weight(
          name = 'boostingOffset',
          shape = list(as.integer(numObs), as.integer(1)),
          initializer = keras::initializer_constant(cdbl$boostingOffset),
          trainable = FALSE
        )
        # self$epsilon <- self$add_weight(
        #   name = 'epsilon', 
        #   shape = list(as.integer(1)),
        #   initializer = keras::initializer_constant(cdbl$epsilon),
        #   trainable = FALSE
        # )
      },
      
      call = forwardPass,
      
      compute_output_shape = function(input_shape) {
        list(input_shape[[1]], self$output_dim)
      }
    )
  )
  
  .trainKeras = function(lr, epochs) {
    # define layer wrapper function
    codacoreLayer <- function(object) {
      keras::create_layer(CustomLayer, object)
    }
    
    # use it in a model
    model <- keras::keras_model_sequential()
    model %>% codacoreLayer()
    if (cdbl$objective == "binary classification") {
      model %>% layer_activation('sigmoid')
    }
    
    # compile graph
    model %>% keras::compile(
      loss = loss_func,
      optimizer = keras::optimizer_sgd(lr, momentum=cdbl$optParams$momentum),
      # optimizer = keras::optimizer_adam(0.001),
      metrics = metrics
    )
    
    
    model %>% keras::fit(cdbl$x, cdbl$y, epochs=epochs, 
                         batch_size=cdbl$optParams$batchSize, 
                         verbose=FALSE)# =TRUE) for debugging
    return(model)
  }
  
  runAdaptively = is.numeric(cdbl$optParams$adaptiveLR) & is.null(cdbl$optParams$vanillaLR)
  if (runAdaptively) {
    # Adaptive learning rate here means that we pick the lr s.t.
    # our first gradient step moves the amalWeights out by a specified amount
    model = .trainKeras(1, 1)
    lr = cdbl$optParams$adaptiveLR
    epochs = cdbl$optParams$epochs
    lr = lr / max(abs(as.numeric(model$get_weights()[[1]])))
    model = .trainKeras(lr, epochs)
  } else {
    warning("Using non-adaptive learning rate may hinder optimization.")
    lr = cdbl$optParams$vanillaLR
    epochs = cdbl$optParams$epochs
    model = .trainKeras(lr, epochs)
  }
  
  
  # Save results:
  cdbl$weights = as.numeric(model$get_weights()[[1]])
  cdbl$softAssignment = 2 / (1 + exp(-cdbl$weights)) - 1
  cdbl$intercept = as.numeric(model$get_weights()[[2]])
  cdbl$slope = as.numeric(model$get_weights()[[3]])
  
  # Equalize the largest + and largest - assignment for more 'balanced' balances
  eqRatio = max(cdbl$softAssignment) / min(cdbl$softAssignment) * (-1)
  cdbl$softAssignment[cdbl$softAssignment < 0] = cdbl$softAssignment[cdbl$softAssignment < 0] * eqRatio
  
  endTime = Sys.time()
  if (cdbl$verbose) {
    print('GD time:')
    print(endTime - startTime)
  }
  # cdbl$runTimeGD = endTime - startTime
  
  return(cdbl)
}

# Given a trained softAssignment, which corresponds to running
# the weights through an activation, we find
# the cutoff at which we define our log-ratio
findBestCutoff.CoDaBaseLearner = function(cdbl) {
  if (any(abs(cdbl$softAssignment) > 0.999999)) {
    warning("Large weights encountered in gradient descent;
            vanishing gradients likely.
            Learning rates might need recalibrating - try adaptive rates?")
  }
  
  candidateCutoffs = sort(abs(cdbl$softAssignment), decreasing=TRUE)
  maxCutoffs = cdbl$cvParams$maxCutoffs
  # Start from 2nd since we equalized +ve and -ve; thus neither side will be empty
  candidateCutoffs = candidateCutoffs[2:min(maxCutoffs, length(candidateCutoffs))]
  
  # TODO: re-implement without passing cdbl to harden()
  # and setInterceptAndSlope() to avoid computational overhead
  # from copying data unnecessarily
  
  # Compute the CV scores:
  startTime = Sys.time()
  numFolds = cdbl$cvParams$numFolds
  # Naive way of splitting equally into folds:
  foldIdx = sample(cut(1:length(cdbl$y), breaks=numFolds, labels=FALSE))
  if (cdbl$objective == "binary classification") {
    # Instead we randomize with equal # of case/controls in each fold
    # See discussion on stratified CV in page 204 of He & Ma 2013
    if (sum(cdbl$y) < numFolds | sum(1 - cdbl$y) < numFolds) {
      stop("Insufficient samples from each class available for cross-validation.")
    }
    caseIdx = sample(cut(1:sum(cdbl$y), breaks=numFolds, labels=FALSE))
    controlIdx = sample(cut(1:sum(1 - cdbl$y), breaks=numFolds, labels=FALSE))
    foldIdx[cdbl$y == 1] = caseIdx
    foldIdx[cdbl$y == 0] = controlIdx
  } 
  scores = matrix(nrow=length(candidateCutoffs), ncol=numFolds)
  i = 0
  for (cutoff in candidateCutoffs) {
    i = i + 1
    cdbl = harden.CoDaBaseLearner(cdbl, cutoff)
    for (j in 1:numFolds) {
      cdbl = setInterceptAndSlope.CoDaBaseLearner(cdbl, cdbl$x[foldIdx != j,], cdbl$y[foldIdx != j], cdbl$boostingOffset[foldIdx != j])
      yHat = predict(cdbl, cdbl$x[foldIdx == j,]) + cdbl$boostingOffset[foldIdx == j]
      if (cdbl$objective == "binary classification") {
        ROC = pROC::roc(cdbl$y[foldIdx == j], yHat, quiet=TRUE)
        scores[i, j] = pROC::auc(ROC)
      } else if (cdbl$objective == "regression") {
        scores[i, j] = -sqrt(mean((cdbl$y[foldIdx == j] - yHat)^2))
      }
    }
  }
  # Now implement lambda-SE rule
  means = apply(scores, 1, mean)
  # see eqn 9.2 here https://www.cs.cmu.edu/~psarkar/sds383c_16/lecture9_scribe.pdf
  stds = apply(scores, 1, stats::sd) / sqrt(numFolds)
  lambdaSeRule = max(means) - stds[which.max(means)] * cdbl$lambda
  # oneSdRule = max(means - stds)
  bestCutoff = candidateCutoffs[means >= lambdaSeRule][1]
  # bestCutoff = candidateCutoffs[which.max(scores)]
  
  
  endTime = Sys.time()
  if (cdbl$verbose) {
    print('CV time:')
    print(endTime - startTime)
    xCoor = 2:(length(means) + 1)
    graphics::plot(xCoor, means, ylim=range(c(means-stds, means+stds)))
    graphics::arrows(xCoor, means-stds, xCoor, means+stds, length=0.05, angle=90, code=3)
    graphics::abline(lambdaSeRule, 0)
  }
  
  if (cdbl$objective == "binary classification") {
    baseLineScore = pROC::auc(pROC::roc(cdbl$y, cdbl$boostingOffset, quiet=TRUE))
  } else if (cdbl$objective == "regression") {
    baseLineScore = -sqrt(mean((cdbl$y - cdbl$boostingOffset)^2))
  }
  noImprovement = lambdaSeRule < baseLineScore
  if (noImprovement) {
    bestCutoff = 1.1 # bigger than the softAssignment
  }
  
  return(bestCutoff)
}


harden.CoDaBaseLearner = function(cdbl, cutoff) {
  numPart = cdbl$softAssignment >= cutoff
  denPart = cdbl$softAssignment <= -cutoff
  hard = list(numerator=numPart, denominator=denPart)
  cdbl$hard = hard
  return(cdbl)
}


setInterceptAndSlope.CoDaBaseLearner = function(cdbl, x, y, boostingOffset) {
  # If our base learner is empty (i.e. couldn't beat the 1SE rule),
  # we simply set to 0:
  if (!any(cdbl$hard$numerator) & !any(cdbl$hard$denominator)) {
    cdbl$slope = 0.0
    cdbl$intercept = 0.0
    return(cdbl)
  }
  # Otherwise, we have a non-empty SLR, so we compute it's regression coefficient
  logRatio = computeLogRatio.CoDaBaseLearner(cdbl, x)
  dat = data.frame(x=logRatio, y=y)
  if (cdbl$objective == "binary classification") {
    glm = stats::glm(y~x, family='binomial', data=dat, offset=boostingOffset)
    if (any(is.na(glm$coefficients))) {
      glm = list(coefficients=list(0, 0))
      warning("Numerical error during glm fit. Possible data issue.") 
    }
  } else if (cdbl$objective == "regression") {
    glm = stats::glm(y~x, family='gaussian', data=dat, offset=boostingOffset)
  } else {
    stop("Not implemented objective=", cdbl$objective)
  }
  cdbl$intercept = glm$coefficients[[1]]
  cdbl$slope = glm$coefficients[[2]]
  return(cdbl)
}


computeLogRatio.CoDaBaseLearner = function(cdbl, x) {
  
  if (!any(cdbl$hard$numerator) | !any(cdbl$hard$denominator)) {
    logRatio = rowSums(x * 0)
  } else { # we have a bona fide log-ratio
    if (cdbl$logRatioType == 'A') {
      epsilon = cdbl$optParams$epsilonA
      pvePart = rowSums(x[, cdbl$hard$numerator, drop=FALSE]) # drop=FALSE to keep as matrix
      nvePart = rowSums(x[, cdbl$hard$denominator, drop=FALSE])
      logRatio = log(pvePart + epsilon) - log(nvePart + epsilon)
    } else if (cdbl$logRatioType == 'B') {
      pvePart = rowMeans(log(x[, cdbl$hard$numerator, drop=FALSE])) # drop=FALSE to keep as matrix
      nvePart = rowMeans(log(x[, cdbl$hard$denominator, drop=FALSE]))
      logRatio = pvePart - nvePart
    }
  }
  
  return(logRatio)
}


predict.CoDaBaseLearner = function(cdbl, x, asLogits=TRUE) {
  logRatio = computeLogRatio.CoDaBaseLearner(cdbl, x)
  eta = cdbl$slope * logRatio + cdbl$intercept
  if (asLogits) {
    return(eta)
  } else {
    if (cdbl$objective == 'regression') {
      stop("Logits argument should only be used for classification, not regression.")
    }
    return(1 / (1 + exp(-eta)))
  }
}


#' codacore
#' 
#' This function implements the codacore algorithm described by Gordon-Rodriguez et al. 2021 
#' (https://doi.org/10.1101/2021.02.11.430695).
#' 
#' @param x A data.frame or matrix of the compositional predictor variables.
#'  Rows represent observations and columns represent variables.
#' @param y A data.frame, matrix or vector of the response. In the case of a 
#'  data.frame or matrix, there should be one row for each observation, and
#'  just a single column.
#' @param logRatioType A string indicating whether to use "balances" or "amalgamations".
#'  Also accepts "balance", "B", "ILR", or "amalgam", "A", "SLR".
#'  Note that the current implementation for balances is not strictly an ILR,
#'  but rather just a collection of balances (which are possibly non-orthogonal
#'  in the Aitchison sense).
#' @param objective A string indicating "binary classification" or "regression". By default,
#'  it is NULL and gets inferred from the values in y.
#' @param lambda A numeric. Corresponds to the "lambda-SE" rule. Sets the "regularization strength"
#'  used by the algorithm to decide how to harden the ratio. 
#'  Larger numbers tend to yield fewer, more sparse ratios.
#' @param offset A numeric vector of the same length as y. Works similarly to the offset in a glm.
#' @param shrinkage A numeric. Shrinkage factor applied to each base learner.
#'  Defaults to 1.0, i.e., no shrinkage applied.
#' @param maxBaseLearners An integer. The maximum number of log-ratios that the model will
#'  learn before stopping. Automatic stopping based on \code{seRule} may occur sooner.
#' @param optParams A list of named parameters for the optimization of the
#'  continuous relaxation. Empty by default. User can override as few or as
#'  many of our defaults as desired. Includes adaptiveLR (learning rate under
#'  adaptive training scheme), momentum (in the gradient-descent sense), 
#'  epochs (number of gradient-descent epochs), batchSize (number of 
#'  observations per minibatch, by default the entire dataset),
#'  and vanillaLR (the learning rate to be used if the user does *not* want
#'  to use the 'adaptiveLR', to be used at the risk of optimization issues).
#' @param cvParams A list of named parameters for the "hardening" procedure
#'  using cross-validation. Includes numFolds (number of folds, default=5) and
#'  maxCutoffs (number of candidate cutoff values of 'c' to be tested out
#'  during CV process, default=20 meaning log-ratios with up to 21 components
#'  can be found by codacore).
#' @param verbose A boolean. Toggles whether to display intermediate steps.
#' @param overlap A boolean. Toggles whether successive log-ratios found by 
#'  CoDaCoRe may contain repeated input variables. TRUE by default.
#'  Changing to FALSE implies that the log-ratios obtained by CoDaCoRe
#'  will become orthogonal in the Aitchison sense, analogously to the
#'  isometric-log-ratio transformation, while losing a small amount of
#'  model flexibility.
#' @param fast A boolean. Whether to run in fast or slow mode. TRUE by
#'  default. Running in slow mode will take ~x5 the computation time,
#'  but may help identify slightly more accurate log-ratios.
#' 
#' @return A \code{codacore} object.
#' 
#' @examples
#' \dontrun{
#' data("Crohn")
#' x <- Crohn[, -ncol(Crohn)]
#' y <- Crohn[, ncol(Crohn)]
#' x <- x + 1
#' model = codacore(x, y)
#' print(model)
#' plot(model)
#' }
#' 
#' @importFrom stats predict
#' 
#' @export
codacore <- function(
  x,
  y,
  logRatioType='balances',
  objective=NULL,
  lambda=1.0,
  offset=NULL,
  shrinkage=1.0,
  maxBaseLearners=5,
  optParams=list(),
  cvParams=list(),
  verbose=FALSE,
  overlap=TRUE,
  fast=TRUE
){
  
  # Convert x and y to the appropriate objects
  x = .prepx(x)
  y = .prepy(y)
  
  # Check whether we are in regression or classification mode by inspecting y
  if (is.null(objective)) {
    distinct_values = length(unique(y))
    if (distinct_values == 2) {
      objective = 'binary classification'
    } else if (inherits(y, 'factor')) {
      stop("Multi-class classification note yet implemented.")
    } else if (inherits(y, 'numeric')) {
      objective = 'regression'
      if (distinct_values <= 10) {
        warning("Response only has ", distinct_values, " distinct values.")
        warning("Consider changing the objective function.")
      }
    }
  }
  
  # Make sure we recognize objective
  if (! objective %in% c('binary classification', 'regression')) {
    stop("Objective: ", objective, " not yet implemented.")
  }
  
  # Save names of labels if relevant
  if (objective == 'binary classification' & inherits(y, 'factor')) {
    yLevels = levels(y)
    y = as.numeric(y) - 1
  } else {
    yLevels = NULL
  }
  
  # In the regression case, standardize data and save scale
  if (objective == 'regression') {
    yMean = mean(y)
    yScale = stats::sd(y)
    y = (y - yMean) / yScale
  } else {
    yMean = NULL
    yScale = NULL
  }
  
  # Convert logRatioType to a unique label:
  if (logRatioType %in% c('amalgamations', 'amalgam', 'A', 'SLR')) {
    logRatioType='A'
  } else if (logRatioType %in% c('balances', 'balance', 'B', 'ILR')) {
    logRatioType='B'
  } else {
    stop('Invalid logRatioType argument given: ', logRatioType)
  }
  
  if (any(x == 0)) {
    if (logRatioType == 'A') {
      warning("The data contain zeros. An epsilon is used to prevent divide-by-zero errors.")
    } else if (logRatioType == 'B') {
      stop("The data contain zeros. Balances cannot be used in this case.")
    }
  }
  
  if (!overlap) {
    # We store away the original data, since we will override during
    # the stagewise-additive procedure, zeroing out the input variables
    # that get picked up by each log-ratio.
    xOriginal = x
  }
  
  if (nrow(x) > 10000) {
    warning("Large number of observations; codacore could benefit from minibatching.")
  }
    
  if (nrow(x) < 50) {
    warning("Small number of observations; proceed with care (the likelihood of unstable results may increase).")
  }
  
  # Set up optimization parameters
  optDefaults = list(
    epochs=100,
    batchSize=nrow(x),
    vanillaLR=NULL,
    adaptiveLR=0.5,
    momentum=0.9,
    epsilonA=1e-6,
    epsilonB=1e-2
    # initialization = 'zeros'
  )
  # Take the defaults and override with any user-specified params, if given
  for (param in names(optParams)) {
    if (param %in% names(optDefaults)) {
      optDefaults[param] = optParams[param]
    } else {
      stop('Unknown optimization parameter given:', param)
    }
  }
  optParams = optDefaults
  
  # Check whether we are running in fast or slow mode
  if (!fast) {
    message("CoDaCoRe is running in slow mode. Switch to fast=TRUE for ~x5 speedup.")
    optParams$epochs = 1000
  }
  
  # Set up cross-validation parameters
  cvDefaults = list(
    maxCutoffs=20,
    numFolds=5
  )
  # Take the defaults and override with any user-specified params, if given
  for (param in names(cvParams)) {
    if (param %in% names(cvDefaults)) {
      cvDefaults[param] = cvParams[param]
    } else {
      stop('Unknown optimization parameter given:', param)
    }
  }
  cvParams = cvDefaults
  
  
  ### Now we train codacore:
  # Initialize from an empty ensemble
  ensemble = list()
  if (is.null(offset)) {
    boostingOffset = y * 0.0
  } else {
    boostingOffset = offset
  }
  maxBaseLearners = maxBaseLearners / shrinkage
  for (i in 1:maxBaseLearners) {
    startTime = Sys.time()
    cdbl = .CoDaBaseLearner(
      x=x,
      y=y,
      boostingOffset=boostingOffset,
      logRatioType=logRatioType,
      objective=objective,
      lambda=lambda,
      optParams=optParams,
      cvParams=cvParams,
      verbose=verbose
    )
    endTime = Sys.time()
    
    if (verbose) {
      cat('\n\n\nBase Learner', i)
      cat('\nLog-ratio indexes:')
      cat('\nNumerator =', which(cdbl$hard$numerator))
      cat('\nDenominator =', which(cdbl$hard$denominator))
      if (objective == 'binary classification') {
        cat('\nAccuracy:', cdbl$accuracy)
        cat('\nAUC:', cdbl$AUC)
      } else if (objective == 'regression') {
        cat('\nRMSE', cdbl$RMSE)
      }
      cat('\nTime taken:', endTime - startTime)
    }
    
    # If base learner is empty, we stop (no further gain in CV AUC):
    if (!any(cdbl$hard$numerator) & !any(cdbl$hard$denominator)) {break}
    
    # Add the new base learner to ensemble
    boostingOffset = boostingOffset + shrinkage * predict(cdbl, x)
    ensemble[[i]] = cdbl
    
    # If AUC is ~1, we stop (we separated the training data):
    # Note this won't always get caught by previous check since separability can lead to
    # numerical overflow which throws an error rather than finding an empty base learner
    if (cdbl$objective == 'binary classification' && cdbl$AUC > 0.999) {break}
    if (cdbl$objective == 'regression' && cdbl$Rsquared > 0.999) {break}
    
    # To avoid overlapping log-ratios, we "zero-out" the input variables that have 
    # already been used
    if (!overlap) {
      x[, cdbl$hard$numerator] = min(x)
      x[, cdbl$hard$denominator] = min(x)
    }
  }
  
  if (!overlap) {
    # Replace the original data frame for saving in the object
    x = xOriginal
  }
  
  cdcr = list(
    ensemble=ensemble,
    x = x,
    y = y,
    objective=objective,
    logRatioType=logRatioType,
    lambda=lambda,
    shrinkage=shrinkage,
    maxBaseLearners=maxBaseLearners,
    optParams=optParams,
    cvParams=cvParams,
    overlap=overlap,
    yLevels=yLevels,
    yMean=yMean,
    yScale=yScale
  )
  class(cdcr) = "codacore"
  
  # If no log-ratios were found, suggest reducing regularization strength
  if (length(ensemble) == 0) {
    warning("No predictive log-ratios were found. Consider using lower values of lambda.")
  }
  
  return(cdcr)
}


#' predict
#'
#' @param object A codacore object.
#' @param newx A set of inputs to our model.
#' @param asLogits Whether to return outputs in logit space
#'  (as opposed to probability space). Should always be set
#'  to TRUE for regression with continuous outputs, but can
#'  be toggled for classification problems.
#' @param numLogRatios How many predictive log-ratios to 
#'  include in the prediction. By default, includes the
#'  effects of all log-ratios that were obtained during
#'  training. Setting this parameter to an integer k will
#'  restrict to using only the top k log-ratios in the model.
#' @param ... Not used.
#'
#' @export
predict.codacore = function(object, newx, asLogits=TRUE, numLogRatios=NA, ...) {
  # Throw an error if zeros are present
  if (any(newx == 0)) {
    if (object$logRatioType == 'A') {
      warning("The data contain zeros. An epsilon is used to prevent divide-by-zero errors.")
    } else if (object$logRatioType == 'B') {
      stop("The data contain zeros. Balances cannot be used in this case.")
    }
  }
  
  x = .prepx(newx)
  yHat = rep(0, nrow(x))
  
  if (is.na(numLogRatios)) {
    numLogRatios = length(object$ensemble)
  }
  
  for (i in 1:numLogRatios) {
    cdbl = object$ensemble[[i]]
    yHat = yHat + object$shrinkage * predict(cdbl, x)
  }
  
  if (object$objective == 'binary classification') {
    if (asLogits) {
      return(yHat)
    } else {
      return(1 / (1 + exp(-yHat)))
    }
  } else if (object$objective == 'regression') {
    return(yHat * object$yScale + object$yMean)
  }
}


#' print
#'
#' @param x A codacore object.
#' @param ... Not used.
#'
#' @export
print.codacore = function(x, ...) {
  # TODO: Make this into a table to print all at once
  cat("\nNumber of log-ratios found:", length(x$ensemble))
  if (length(x$ensemble) >= 1) {
    for (i in 1:length(x$ensemble)) {
      cat("\n***")
      cat("\nLog-ratio rank", i)
      cdbl = x$ensemble[[i]]
      hard = x$ensemble[[i]]$hard
      if (is.null(rownames(cdbl$x))) {
        cat("\nNumerator:", which(cdbl$hard$numerator))
        cat("\nDenominator:", which(cdbl$hard$denominator))
      } else {
        cat("\nNumerator:", colnames(cdbl$x)[which(cdbl$hard$numerator)])
        cat("\nDenominator:", colnames(cdbl$x)[which(cdbl$hard$denominator)])
      }
      # cat("\nIntercept:", cdbl$intercept)
      if (cdbl$objective == 'binary classification') {
        cat("\nAUC:", cdbl$AUC)
        cat("\nSlope:", cdbl$slope)
      } else if (cdbl$objective == 'regression') {
        cat("\nR squared:", cdbl$Rsquared)
        cat("\nSlope:", cdbl$slope * x$yScale)
      }
    }
  }
  cat("\n") # one final new line at end to finish print block
}


#' plot
#' 
#' Plots a summary of a fitted codacore model.
#' Credit to the authors of the selbal package (Rivera-Pinto et al., 2018),
#' from whose package these plots were inspired.
#'
#' @param x A codacore object.
#' @param index The index of the log-ratio to plot.
#' @param ... Not used.
#'
#' @export
plot.codacore = function(x, index = 1, ...) {
  
  allRatios = getLogRatios(x)
  if(index > ncol(allRatios)){
    stop("The selected log-ratio does not exist!")
  }
  
  if (x$objective == 'regression') {
    
    logRatio = allRatios[, index]
    graphics::plot(logRatio, x$y, xlab='Log-ratio score', ylab='Response')
    graphics::abline(x$ensemble[[1]]$intercept, x$ensemble[[1]]$slope, lwd=2)
    
  } else if (x$objective == 'binary classification') {
    
    logRatio = allRatios[, index]
    
    # Convert 0/1 binary output to the original labels, if any
    if (!is.null(x$yLevels)) {
      y = x$yLevels[x$y + 1]
    }
    
    graphics::boxplot(
      logRatio ~ y,
      col=c('orange','lightblue'),
      main=paste0('Distribution of log-ratio ', index),
      xlab='Log-ratio score',
      ylab='Outcome',
      horizontal=TRUE
    )
    
  }
}


#' plotROC
#'
#' @param cdcr A codacore object.
#'
#' @export
plotROC = function(cdcr) {
  
  if (cdcr$objective != 'binary classification') {
    stop("ROC curves undefined for binary classification")
  }
  cols = c("black", "gray50", "gray70", "gray80", "gray90")
  lwds = c(2.0, 1.5, 1.2, 0.8, 0.6)
  oldPar <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(oldPar)) # make sure to restore params even if there's an error
  graphics::par(pty = 's')
  graphics::plot(cdcr$ensemble[[1]]$ROC)
  legendCols = cols
  numBL = length(cdcr$ensemble)
  legendText = c()
  legendLwds = c()
  for (i in 1:min(5, numBL)) {
    cdbl = cdcr$ensemble[[i]]
    graphics::lines(cdbl$ROC$specificities, cdbl$ROC$sensitivities, col=cols[i], lwd=lwds[i])
    legendText = c(legendText, paste0("Log-ratio: ", i, ", AUC: ", round(cdbl$AUC, 2)))
    legendCols = c(legendCols, cols[i])
    legendLwds = c(legendLwds, lwds[i])
  }
  graphics::legend(
    "bottomright",
    rev(legendText),
    lty=1,
    col=rev(legendCols),
    lwd=rev(legendLwds) + 0.5
  )
}


# Helper functions below...


#' activeInputs
#'
#' @param cdcr A codacore object.
#'
#' @return The covariates included in the log-ratios
#' 
#' @export
activeInputs.codacore = function(cdcr) {
  
  vars = c()
  
  for (cdbl in cdcr$ensemble) {
    vars = c(vars, which(cdbl$hard$numerator))
    vars = c(vars, which(cdbl$hard$denominator))
  }
  
  return(sort(unique(vars)))
}


#' getNumeratorParts
#'
#' @param cdcr A codacore object.
#' @param baseLearnerIndex An integer indicating which of the 
#'     (possibly multiple) log-ratios learned by codacore to be used.
#' @param boolean Whether to return the parts in boolean form
#'     (a vector of TRUE/FALSE) or to return the column names of
#'     those parts directly.
#'
#' @return The covariates in the numerator of the selected log-ratio.
#' 
#' @export
getNumeratorParts <- function(cdcr, baseLearnerIndex=1, boolean=TRUE){
  
  parts = cdcr$ensemble[[baseLearnerIndex]]$hard$numerator
  
  if (boolean) {
    return(parts)
  } else {
    return(colnames(cdcr$x)[parts])
  }
}

#' getDenominatorParts
#'
#' @param cdcr A codacore object.
#' @param baseLearnerIndex An integer indicating which of the 
#'     (possibly multiple) log-ratios learned by codacore to be used.
#' @param boolean Whether to return the parts in boolean form
#'     (a vector of TRUE/FALSE) or to return the column names of
#'     those parts directly.
#' 
#' @return The covariates in the denominator of the selected log-ratio.
#' 
#' @export
getDenominatorParts <- function(cdcr, baseLearnerIndex=1, boolean=TRUE){
  
  parts = cdcr$ensemble[[baseLearnerIndex]]$hard$denominator
  
  if (boolean) {
    return(parts)
  } else {
    return(colnames(cdcr$x)[parts])
  }
}

#' getLogRatios
#'
#' @param cdcr A codacore object
#' @param x A set of (possibly unseen) compositional data. 
#'     The covariates must be passed in the same order as 
#'     for the original codacore() call.
#'
#' @return The learned log-ratio features, computed on input x.
#' 
#' @export
getLogRatios <- function(cdcr, x=NULL){
  
  if (is.null(x)) {
    x = cdcr$x
  }
  
  if (cdcr$logRatioType == 'A') {
    epsilonA = cdcr$optParams$epsilonA
    ratios <- lapply(cdcr$ensemble, function(a){
      num <- rowSums(x[, a$hard$numerator, drop=FALSE]) + epsilonA
      den <- rowSums(x[, a$hard$denominator, drop=FALSE]) + epsilonA
      log(num/den)
    })
  } else if (cdcr$logRatioType == 'B') {
    ratios <- lapply(cdcr$ensemble, function(a){
      num <- rowMeans(log(x[, a$hard$numerator, drop=FALSE]))
      den <- rowMeans(log(x[, a$hard$denominator, drop=FALSE]))
      num - den
    })
  }
  
  out <- do.call("cbind", ratios)
  colnames(out) <- paste0("log-ratio", 1:ncol(out))
  return(out)
}


#' getSlopes
#'
#' @param cdcr A codacore object
#'
#' @return The slopes (i.e., regression coefficients) for each log-ratio.
#' 
#' @export
getSlopes <- function(cdcr){
  
  out = c()
  
  for (cdbl in cdcr$ensemble) {
    out = c(out, cdbl$slope)
  }
  
  return(out)
}


#' getNumLogRatios
#'
#' @param cdcr A codacore object
#'
#' @return The number of log-ratios that codacore found.
#'     Typically a small integer. Can be zero if codacore
#'     found no predictive log-ratios in the data.
#' 
#' @export
getNumLogRatios <- function(cdcr){
  return(length(cdcr$ensemble))
}


#' getTidyTable
#'
#' @param cdcr A codacore object
#'
#' @return A table displaying the log-ratios found.
#' 
#' @export
getTidyTable <- function(cdcr){
  
  tidyLogRatio = function(baseLearnerIndex, model, xTrain){
    x = getNumeratorParts(model, baseLearnerIndex, FALSE)
    df = data.frame(Side = 'Numerator', Name = x)
    x = getDenominatorParts(model, baseLearnerIndex, FALSE)
    df = rbind(df, data.frame(Side = 'Denominator', Name = x))
    df$logRatioIndex = baseLearnerIndex
    return(df)
  }
  
  num = getNumLogRatios(cdcr)
  
  if (num == 0) {
    return()
  } else {
    do.call(rbind, lapply(1:num, tidyLogRatio, model=cdcr))
  }
}

#' getBinaryPartitions
#'
#' @param cdcr A codacore object
#'
#' @return A matrix describing whether each component (as rows) is found in the
#'  numerator (1) or denominator (-1) of each learned log-ratio (as columns).
#'  This format resembles a serial binary partition matrix frequently used
#'  in balance analysis.
#' 
#' @export
getBinaryPartitions <- function(cdcr){
  
  numBaseLearners <- length(cdcr$ensemble)
  res <- list(numBaseLearners)
  for(baseLearner in 1:numBaseLearners){
    thisNumerator <- getNumeratorParts(cdcr, baseLearner)
    thisDenominater <- getDenominatorParts(cdcr, baseLearner)
    res[[baseLearner]] <- thisNumerator*1 + thisDenominater*-1
  }
  do.call("cbind", res)
}

.prepx = function(x) {
  if (class(x)[1] == 'tbl_df') {x = as.data.frame(x)}
  if (class(x)[1] == 'data.frame') {x = as.matrix(x)}
  if (is.integer(x)) {x = x * 1.0}
  
  # If the data is un-normalized (e.g. raw counts),
  # we normalize it to ensure our learning rate is well calibrated
  x = x / rowSums(x)
  return(x)
}

.prepy = function(y) {
  if (inherits(y, 'tbl_df')) {
    y = as.data.frame(y)
  }
  if (inherits(y, 'data.frame')) {
    if (ncol(y) > 1) {
      stop("Response should be 1-dimensional (if given 
           as a data.frame or matrix, it should have a 
           row for each sample, and a single column).")
    }
    y = y[[1]]
  }
  if (inherits(y, 'matrix')) {
    if (ncol(y) > 1) {
      stop("Response should be 1-dimensional (if given 
           as a data.frame or matrix, it should have a 
           row for each sample, and a single column).")
    }
    if (inherits(y, 'character')) {
      y = as.character(y)
    }
    if (inherits(y, 'numeric')){
      y = as.numeric(y)
    }
  }
  if (inherits(y, 'character')) {
    y = factor(y)
  }
  return(y)
}

