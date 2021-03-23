
## codacore 0.0.1
---------------------
* Fix a bug in lambda-standard-error rule.
    * Estimation of cross-validation prediction error was missing a scaling factor to account for the number of folds.
    * As a result, models were over-regularized.
* Update guide.