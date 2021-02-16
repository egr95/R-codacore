test_that("multiplication works", {
  set.seed(0)
  n = 1000
  p = 100
  HTS = simulateHTS(n, p)
  x = HTS$x + 1
  y = HTS$y
  model = codacore(x, y)
  
  expect_true(getNumeratorParts(model, 1)[1])
  expect_true(getDenominatorParts(model, 1)[2])
})
