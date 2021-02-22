
# Here we implement the codacore model

library(keras)

# """Fits a single base learner"""
# Private class not to be called by user
.CoDaBaseLearner <- function(
  x,
  y,
  boostingOffset,
  type,
  mode,
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
    ROC=NULL,
    x=x,
    y=y,
    boostingOffset=boostingOffset,
    type=type,
    mode=mode,
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
  cdbl$ROC = pROC::roc(y, yHat, quiet=T)
  cdbl$AUC = pROC::auc(cdbl$ROC)
  cdbl$accuracy = mean(y == (yHat > 0))
  
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
  if (cdbl$mode == "classification") {
    loss_func = 'binary_crossentropy'
    if (abs(mean(1 / (1 + exp(-cdbl$boostingOffset))) - mean(cdbl$y)) < 0.001) {
      # Protect against numerical errors in glm() call
      interceptInit = 0.0
    } else {
      tempGLM = stats::glm(cdbl$y ~ 1, offset=cdbl$boostingOffset, family='binomial')
      interceptInit = tempGLM$coef[[1]]
    }
  } else if (cdbl$mode == "regression") {
    loss_func = 'mean_squared_error'
    interceptInit = mean(cdbl$y - cdbl$boostingOffset)
  }
  
  # Define the forward pass for our relaxation,
  # which differs for balances and amalgamations
  if (cdbl$type == 'A') {
    epsilon = cdbl$optParams$epsilonA
    forwardPass = function(x, mask = NULL) {
      softAssignment = 2 * keras::k_sigmoid(self$weights) - 1
      # Add the small value to ensure gradient flows at exact zeros (initial values)
      pvePart = keras::k_dot(x, keras::k_relu(softAssignment + 1e-20))
      nvePart = keras::k_dot(x, keras::k_relu(-softAssignment))
      logRatio = keras::k_log(pvePart + epsilon) - 
        keras::k_log(nvePart + epsilon)
      eta = self$slope * logRatio + self$intercept + self$boostingOffset
      keras::k_sigmoid(eta)
    }
  } else if (cdbl$type == 'B') {
    epsilon = cdbl$optParams$epsilonB
    forwardPass = function(x, mask = NULL) {
      softAssignment = 2 * keras::k_sigmoid(self$weights) - 1
      # Add the small value to ensure gradient flows at exact zeros (initial values)
      pvePart = keras::k_relu(softAssignment + 1e-20)
      nvePart = keras::k_relu(-softAssignment)
      logRatio = keras::k_dot(keras::k_log(x), pvePart) / keras::k_maximum(keras::k_sum(pvePart), epsilon) -
        keras::k_dot(keras::k_log(x), nvePart) / keras::k_maximum(keras::k_sum(nvePart), epsilon)
      eta = self$slope * logRatio + self$intercept + self$boostingOffset
      keras::k_sigmoid(eta)
    }
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
          initializer = keras::initializer_constant(0.1),
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
    
    # compile graph
    model %>% keras::compile(
      loss = loss_func,
      optimizer = keras::optimizer_sgd(lr=lr, momentum=cdbl$optParams$momentum),
      # optimizer = keras::optimizer_adam(lr=0.001),
      metrics = c('accuracy')
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
  
  candidateCutoffs = sort(abs(cdbl$softAssignment), decreasing=T)
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
  foldIdx = sample(cut(1:length(cdbl$y), breaks=numFolds, labels=F))
  # Instead we randomize with equal # of case/controls in each fold
  # See discussion on stratified CV in page 204 of He & Ma 2013
  caseIdx = sample(cut(1:sum(cdbl$y), breaks=numFolds, labels=F))
  controlIdx = sample(cut(1:sum(1 - cdbl$y), breaks=numFolds, labels=F))
  foldIdx[cdbl$y == 1] = caseIdx
  foldIdx[cdbl$y == 0] = controlIdx
  scores = matrix(nrow=length(candidateCutoffs), ncol=numFolds)
  i = 0
  for (cutoff in candidateCutoffs) {
    i = i + 1
    cdbl = harden.CoDaBaseLearner(cdbl, cutoff)
    for (j in 1:numFolds) {
      cdbl = setInterceptAndSlope.CoDaBaseLearner(cdbl, cdbl$x[foldIdx != j,], cdbl$y[foldIdx != j], cdbl$boostingOffset[foldIdx != j])
      yHat = predict(cdbl, cdbl$x[foldIdx == j,]) + cdbl$boostingOffset[foldIdx == j]
      ROC = pROC::roc(cdbl$y[foldIdx == j], yHat, quiet=T)
      scores[i, j] = pROC::auc(ROC)
    }
  }
  # Now implement lambda-SE rule
  means = apply(scores, 1, mean)
  stds = apply(scores, 1, stats::sd)
  lambdaSeRule = max(means) - stds[which.max(means)] * cdbl$lambda
  # oneSdRule = max(means - stds)
  bestCutoff = candidateCutoffs[means >= lambdaSeRule][1]
  # bestCutoff = candidateCutoffs[which.max(scores)]
  
  
  endTime = Sys.time()
  if (cdbl$verbose) {
    print('CV time:')
    print(endTime - startTime)
    graphics::plot(2:maxCutoffs, means, ylim=range(c(means-stds, means+stds)))
    graphics::arrows(2:maxCutoffs, means-stds, 2:maxCutoffs, means+stds, length=0.05, angle=90, code=3)
    graphics::abline(lambdaSeRule, 0)
  }
  
  noImprovement = lambdaSeRule < pROC::auc(pROC::roc(cdbl$y, cdbl$boostingOffset, quiet=T))
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
  if (cdbl$mode == "classification") {
    glm = stats::glm(y~x, family='binomial', data=dat, offset=boostingOffset)
  } else if (cdbl$mode == "regression") {
    warning("Not tested")
    glm = stats::glm(y~x, family='normal', data=dat, offset=boostingOffset)
  } else {
    stop("Not implemented mode=", cdbl$mode)
  }
  cdbl$intercept = glm$coefficients[[1]]
  cdbl$slope = glm$coefficients[[2]]
  return(cdbl)
}


computeLogRatio.CoDaBaseLearner = function(cdbl, x) {
  
  if (!any(cdbl$hard$numerator) | !any(cdbl$hard$denominator)) {
    logRatio = rowSums(x * 0)
  } else { # we have a bona fide log-ratio
    if (cdbl$type == 'A') {
      epsilon = cdbl$optParams$epsilonA
      pvePart = rowSums(x[, cdbl$hard$numerator, drop=F]) # drop=F to keep as matrix
      nvePart = rowSums(x[, cdbl$hard$denominator, drop=F])
      logRatio = log(pvePart + epsilon) - log(nvePart + epsilon)
    } else if (cdbl$type == 'B') {
      pvePart = rowMeans(log(x[, cdbl$hard$numerator, drop=F])) # drop=F to keep as matrix
      nvePart = rowMeans(log(x[, cdbl$hard$denominator, drop=F]))
      logRatio = pvePart - nvePart
    }
  }
  
  return(logRatio)
}


predict.CoDaBaseLearner = function(cdbl, x, logits=T) {
  logRatio = computeLogRatio.CoDaBaseLearner(cdbl, x)
  eta = cdbl$slope * logRatio + cdbl$intercept
  if (logits) {
    return(eta)
  } else {
    return(1 / (1 + exp(-eta)))
  }
}


#' codacore
#' 
#' This function implements the codacore algorithm described in https://doi.org/10.1101/2021.02.11.430695.
#' 
#' @param x A data.frame of the compositional predictor variables.
#' @param y A data.frame of the response variables.
#' @param type A string indicating whether to use "balances" or "amalgamations".
#' Also accepts "balance", "B", "ILR", or "amalgam", "A", "SLR".
#' Note that the current implementation for balances is not strictly an ILR,
#' but rather just a collection of balances (which are possibly non-orthogonal
#' in the Aitchison sense).
#' @param mode A string indicating "classification" or "regression".
#' @param lambda A numeric. Corresponds to the "lambda-SE" rule. Sets the "regularization strength"
#'  used by the algorithm to decide how to harden the ratio. 
#'  Larger numbers tend to yield fewer, more sparse ratios.
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
#'  using cross-validation. Includes numFolds (number of folds) and
#'  maxCutoffs (number of candidate cutoff values of 'c' to be tested out
#'  during CV process).
#' @param verbose A logical. Toggles whether to display intermediate steps.
#' @param overlap TODO: To be implemented
#' 
#' @importFrom stats predict
#' 
#' @export
codacore <- function(
  x,
  y,
  type='balances',
  mode='classification',
  lambda=1.0,
  shrinkage=1.0,
  maxBaseLearners=10,
  optParams=list(),
  cvParams=list(),
  verbose=F,
  overlap=T #TODO: implement this or remove
){
  
  # Convert x and y to the appropriate objects
  x = .prepx(x)
  y = .prepy(y)

  # Convert type to a unique label:
  if (type %in% c('amalgamations', 'amalgam', 'A', 'SLR')) {
    type='A'
  } else if (type %in% c('balances', 'balance', 'B', 'ILR')) {
    type='B'
  } else {
    stop('Invalid type argument given: ', type)
  }
  
  if (any(x == 0)) {
    if (type == 'A') {
      warning("The data contain zeros. An epsilon is used to prevent divide-by-zero errors.")
    } else if (type == 'B') {
      stop("The data contain zeros. Balances cannot be used in this case.")
    }
  }
  
  if (mode != 'classification') {
    stop("Mode: ", mode, " not yet implemented.")
  }
  
  if (!overlap) {
    stop("Disjoint log-ratios not yet implemented.")
  }
  
  if (nrow(x) > 10000) {
    warning("Large number of observations; codacore could benefit from minibatching.")
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
  boostingOffset = y * 0.0
  maxBaseLearners = maxBaseLearners / shrinkage
  for (i in 1:maxBaseLearners) {
    startTime = Sys.time()
    cdbl = .CoDaBaseLearner(
      x=x,
      y=y,
      boostingOffset=boostingOffset,
      type=type,
      mode=mode,
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
      cat('\nAccuracy:', cdbl$accuracy)
      cat('\nAUC:', cdbl$AUC)
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
    if (cdbl$AUC > 0.999) {break}
  }
  
  cdcr = list(
    ensemble=ensemble,
    x = x,
    y = y,
    mode=mode,
    type=type,
    lambda=lambda,
    shrinkage=shrinkage,
    maxBaseLearners=maxBaseLearners,
    optParams=optParams,
    cvParams=cvParams
  )
  class(cdcr) = "codacore"
  
  return(cdcr)
}


predict.codacore = function(cdcr, x, logits=T) {
  x = .prepx(x)
  yHat = rep(0, nrow(x))
  for (cdbl in cdcr$ensemble) {
    yHat = yHat + cdcr$shrinkage * predict(cdbl, x)
  }
  if (logits) {
    return(yHat)
  } else {
    return(1 / (1 + exp(-yHat)))
  }
}


#' print
#'
#' @param x A codacore object.
#' @param ... Not used.
#'
#' @return
#' @export
#'
#' @examples
print.codacore = function(x, ...) {
  # TODO: Make this into a table to print all at once
  cat("\nNumber of base learners found:", length(x$ensemble))
  for (i in 1:length(x$ensemble)) {
    cat("\n***")
    cat("\nBase Learner", i)
    cdbl = x$ensemble[[i]]
    hard = x$ensemble[[i]]$hard
    cat("\nNumerator:", which(cdbl$hard$numerator))
    cat("\nDenominator:", which(cdbl$hard$denominator))
    cat("\nIntercept:", cdbl$intercept)
    cat("\nSlope:", cdbl$slope)
    cat("\nAUC:", cdbl$AUC)
  }
  cat("\n") # one final new line at end to finish print block
}


#' plot
#'
#' @param x A codacore object.
#' @param ... Not used.
#'
#' @return
#' @export
#'
#' @examples
plot.codacore = function(x, ...) {
  cols = c("black", "gray40", "gray60", "gray80")
  lwds = c(2.0, 1.5, 1.2, 0.8)
  graphics::plot(x$ensemble[[1]]$ROC)
  for (i in 2:min(4, length(x$ensemble))) {
    graphics::lines(x$ensemble[[i]]$ROC$specificities, x$ensemble[[i]]$ROC$sensitivities, col=cols[i], lwd=lwds[i])
  }
  graphics::legend(
    "bottomright",
     c("Base Learner 4", "Base Learner 3", "Base Learner 2", "Base Learner 1"),
     # fill=c("gray90", "gray80", "gray50", "black"),
     lty=1,
     # bty='n'
     col=rev(cols),
     lwd=rev(lwds) + 0.5
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
#'
#' @return The covariates in the numerator of the selected log-ratio.
#' 
#' @export
getNumeratorParts <- function(cdcr, baseLearnerIndex = 1){
  
  cdcr$ensemble[[baseLearnerIndex]]$hard$numerator
}

#' getDenominatorParts
#'
#' @param cdcr A codacore object.
#' @param baseLearnerIndex An integer indicating which of the 
#'     (possibly multiple) log-ratios learned by codacore to be used.
#'
#' @return The covariates in the denominator of the selected log-ratio.
#' 
#' @export
getDenominatorParts <- function(cdcr, baseLearnerIndex = 1){
  
  cdcr$ensemble[[baseLearnerIndex]]$hard$denominator
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
  
  if (cdcr$type == 'A') {
    epsilonA = cdcr$optParams$epsilonA
    ratios <- lapply(cdcr$ensemble, function(a){
      num <- rowSums(x[, a$hard$numerator, drop=FALSE]) + epsilonA
      den <- rowSums(x[, a$hard$denominator, drop=FALSE]) + epsilonA
      log(num/den)
    })
  } else if (cdcr$type == 'B') {
    ratios <- lapply(cdcr$ensemble, function(a){
      num <- rowMeans(log(x[, a$hard$numerator, drop=FALSE]))
      den <- rowMeans(log(x[, a$hard$denominator, drop=FALSE]))
      log(num/den)
    })
  }
  
  out <- do.call("cbind", ratios)
  colnames(out) <- paste0("log-ratio", 1:ncol(out))
  return(out)
}

.prepx = function(x) {
  if (class(x) == 'data.frame') {x = as.matrix(x)}
  if (is.integer(x)) {x = x * 1.0}
  
  # If the data is un-normalized (e.g. raw counts),
  # we normalize it to ensure our learning rate is well calibrated
  x = x / rowSums(x)
  return(x)
}

.prepy = function(y) {
  if (class(y) == 'data.frame') {y = y[[1]]}
  if (class(y) == 'factor') {y = as.numeric(y) - 1}
  return(y)
}