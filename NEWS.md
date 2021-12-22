## codacore 0.0.3
---------------------
* Submitted to CRAN.
* Updated readme and vignettes to reflect this.

## codacore 0.0.2
---------------------
* Updated tests.
* Updated guide.
    * Covariate adjustment
    * Unsupervised learning
    * Multi-omics
* Minor bugfix with glm numerics.
* Added numLogRatios param to predict().

## codacore 0.0.1
---------------------
* Fix a bug in lambda-standard-error rule.
    * Estimation of cross-validation prediction error was missing a scaling factor to account for the number of folds.
    * As a result, models were over-regularized.
* Update guide.