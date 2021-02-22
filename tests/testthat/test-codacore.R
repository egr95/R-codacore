testthat::test_that("simple logratios", {
  set.seed(0)
  tensorflow::set_random_seed(0)
  n = 1000
  p = 100
  HTS = simulateHTS(n, p)
  x = HTS$x + 1
  y = HTS$y
  model = codacore(x, y, type='B')
  testthat::expect_true(getNumeratorParts(model, 1)[1])
  testthat::expect_true(getDenominatorParts(model, 1)[2])
  testthat::expect_equal(model$ensemble[[1]]$accuracy, 0.846)
  
  model = codacore(x, y, type='A')
  testthat::expect_true(getNumeratorParts(model, 1)[1])
  testthat::expect_true(getDenominatorParts(model, 1)[2])
  testthat::expect_equal(model$ensemble[[1]]$accuracy, 0.846)
})

testthat::test_that("balances", {
  set.seed(0)
  tensorflow::set_random_seed(0)
  n = 1000
  p = 100
  HTS = simulateHTS(n, p, logratio='balance')
  x = HTS$x + 1
  y = HTS$y
  model = codacore(x, y, type='B')
  
  testthat::expect_true(getNumeratorParts(model, 1)[4])
  testthat::expect_true(getNumeratorParts(model, 1)[6])
  testthat::expect_true(getDenominatorParts(model, 1)[5])
  testthat::expect_equal(model$ensemble[[1]]$accuracy, 0.73)
})

testthat::test_that("amalgamations", {
  set.seed(0)
  tensorflow::set_random_seed(0)
  n = 1000
  p = 100
  HTS = simulateHTS(n, p, logratio='amalgamation')
  x = HTS$x + 1
  y = HTS$y
  model = codacore(x, y, type='A')
  
  testthat::expect_true(getNumeratorParts(model, 1)[1])
  testthat::expect_true(getNumeratorParts(model, 1)[2])
  testthat::expect_true(getDenominatorParts(model, 1)[3])
  testthat::expect_equal(model$ensemble[[1]]$AUC[1], 0.916, tolerance=0.001)
})

