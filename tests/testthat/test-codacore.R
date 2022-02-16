tensorflow_is_installed <- function(){
  check <- tryCatch({
    # dummy tensorflow code
    tensorflow::set_random_seed(0)
    TRUE
  }, error = function(e){
    FALSE
  })
  return(check)
}

testthat::test_that("simple logratios", {
  if (tensorflow_is_installed()){
    set.seed(0)
    tensorflow::set_random_seed(0)
    n = 1000
    p = 100
    HTS = simulateHTS(n, p)
    x = HTS$x + 1
    y = HTS$y
    model = codacore(x, y, logRatioType='B')
    testthat::expect_true(getNumeratorParts(model, 1)[1])
    testthat::expect_true(getDenominatorParts(model, 1)[2])
    testthat::expect_equal(model$ensemble[[1]]$accuracy, 0.851)
    
    model = codacore(x, y, logRatioType='A')
    testthat::expect_true(getNumeratorParts(model, 1)[1])
    testthat::expect_true(getDenominatorParts(model, 1)[2])
    testthat::expect_equal(model$ensemble[[1]]$accuracy, 0.846)
    
    # test getBinaryPartitions() function
    testthat::expect_true(getBinaryPartitions(model)[1,1] == 1)
    testthat::expect_true(getBinaryPartitions(model)[2,1] == -1)
    testthat::expect_true(getBinaryPartitions(model)[3,1] == 0)
    
    # Now test in regression mode
    HTS = simulateHTS(n, p, outputType = 'continuous')
    x = HTS$x + 1
    y = HTS$y
    model = codacore(x, y, logRatioType='B', objective='regression')
    testthat::expect_true(getNumeratorParts(model, 1)[1])
    testthat::expect_true(getDenominatorParts(model, 1)[2])
    testthat::expect_equal(model$ensemble[[1]]$Rsquared, 0.349, tolerance=0.001)
    
    model = codacore(x, y, logRatioType='A', objective='regression')
    testthat::expect_true(getNumeratorParts(model, 1)[1])
    testthat::expect_true(getDenominatorParts(model, 1)[2])
    testthat::expect_equal(model$ensemble[[1]]$Rsquared, 0.349, tolerance=0.001)
  }
})

testthat::test_that("balances", {
  if (tensorflow_is_installed()){
    set.seed(0)
    tensorflow::set_random_seed(0)
    n = 1000
    p = 100
    HTS = simulateHTS(n, p, logratio='balance')
    x = HTS$x + 1
    y = HTS$y
    model = codacore(x, y, logRatioType='B')
    
    testthat::expect_true(getNumeratorParts(model, 1)[4])
    testthat::expect_true(getNumeratorParts(model, 1)[6])
    testthat::expect_true(getDenominatorParts(model, 1)[5])
    testthat::expect_equal(model$ensemble[[1]]$accuracy, 0.733)
    
    # Now test in regression mode
    HTS = simulateHTS(n, p, logratio='balance', outputType = 'continuous')
    x = HTS$x + 1
    y = HTS$y
    model = codacore(x, y, logRatioType='B', objective='regression')
    testthat::expect_equal(model$ensemble[[1]]$Rsquared, 0.257, tolerance=0.001)
  }
})

testthat::test_that("amalgamations", {
  if (tensorflow_is_installed()){
    set.seed(0)
    tensorflow::set_random_seed(0)
    n = 1000
    p = 100
    HTS = simulateHTS(n, p, logratio='amalgamation')
    x = HTS$x + 1
    y = HTS$y
    model = codacore(x, y, logRatioType='A')
    
    testthat::expect_true(getNumeratorParts(model, 1)[1])
    testthat::expect_true(getNumeratorParts(model, 1)[2])
    testthat::expect_true(getDenominatorParts(model, 1)[3])
    testthat::expect_equal(model$ensemble[[1]]$AUC[1], 0.925, tolerance=0.001)
    
    
    # Now test in regression mode
    HTS = simulateHTS(n, p, logratio='amalgamation', outputType = 'continuous')
    x = HTS$x + 1
    y = HTS$y
    model = codacore(x, y, logRatioType='A', objective='regression')
    testthat::expect_true(getNumeratorParts(model, 1)[1])
    testthat::expect_true(getNumeratorParts(model, 1)[2])
    testthat::expect_true(getDenominatorParts(model, 1)[3])
    testthat::expect_equal(model$ensemble[[1]]$Rsquared, 0.540, tolerance=0.001)
  }
})

