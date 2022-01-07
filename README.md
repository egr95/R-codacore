# R-codacore

A self-contained, up-to-date implementation of [CoDaCoRe](https://doi.org/10.1093/bioinformatics/btab645), in the R programming language, by the original authors.

The [CoDaCoRe guide](https://egr95.github.io/R-codacore/inst/misc/guide.html) contains a detailed tutorial on installation, usage and functionality.

Note this repository is under active development. If you would like to use CoDaCoRe on your dataset, and have any questions regarding the installation, usage, implementation, or model itself, do not hesitate to contact <eg2912@columbia.edu>. Some previously asked questions are available on the [Issues page](https://github.com/egr95/R-codacore/issues).
Contributions, fixes, and feature requests are also welcome - please create an issue, submit a pull request, or email me.

## Quick-start: how to install and run CoDaCoRe

*Update: [CoDaCoRe is now live on CRAN](https://cran.r-project.org/web/packages/codacore/index.html)!*

1. We can install CoDaCoRe by running (further details in the [guide](https://egr95.github.io/R-codacore/inst/misc/guide.html#installation)):

```r
install.packages('codacore')
```

2. To fit codacore on some data and check the results (further details in the [guide](https://egr95.github.io/R-codacore/inst/misc/guide.html#training-the-model):
```r
library("codacore")
help(codacore) # if in doubt, check documentation
data("Crohn") # load some data and apply codacore
x <- Crohn[, -ncol(Crohn)] + 1
y <- Crohn[, ncol(Crohn)]
model = codacore(
    x, # compositional input, e.g., HTS count data 
    y, # response variable, typically a 0/1 binary indicator 
    logRatioType = "balances", # can use "amalgamations" instead, or abbreviations "B" and "A"
    lambda = 1 # regularization strength (default corresponds to 1SE rule) 
)
print(model)
plot(model)
```

## Acknowledgements
Thanks for your contributions to R-codacore!

- Marcus Fedarko
- Gregor Seyer
