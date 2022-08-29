[![CRAN_Status_Badge](https://cranlogs.r-pkg.org:443/badges/codacore)](https://cranlogs.r-pkg.org:443/badges/codacore)
[![Downloads](https://cranlogs.r-pkg.org:443/badges/codacore)](https://cranlogs.r-pkg.org:443/badges/codacore)
[![Total Downloads](https://cranlogs.r-pkg.org:443/badges/grand-total/codacore)](https://cranlogs.r-pkg.org:443/badges/grand-total/codacore)

# codacore

*Update: [CoDaCoRe is now live on CRAN](https://CRAN.R-project.org/package=codacore/)*

A self-contained, up-to-date implementation of [CoDaCoRe](https://doi.org/10.1093/bioinformatics/btab645), in the R programming language, by the original authors.

The [CoDaCoRe guide](https://egr95.github.io/R-codacore/inst/misc/guide.html) contains a detailed tutorial on installation, usage and functionality.

Note this repository is under active development. If you would like to use CoDaCoRe on your dataset, and have any questions regarding the installation, usage, implementation, or model itself, do not hesitate to contact <eg2912@columbia.edu>. Some previously asked questions are available on the [Issues page](https://github.com/egr95/R-codacore/issues).
Contributions, fixes, and feature requests are also welcome - please create an issue, submit a pull request, or email me.

## Quick-start: how to install and run CoDaCoRe

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

## Reference

Gordon-Rodriguez, Elliott, Thomas P. Quinn, and John P. Cunningham. "Learning sparse log-ratios for high-throughput sequencing data." Bioinformatics 38.1 (2022): 157-163. [[link](https://doi.org/10.1093/bioinformatics/btab645)]

Quinn, Thomas P., Elliott Gordon-Rodriguez, and Ionas Erb. "A critique of differential abundance analysis, and advocacy for an alternative." arXiv preprint arXiv:2104.07266 (2021). [[link](https://arxiv.org/abs/2104.07266)]

## Acknowledgements
Thanks for your contributions to codacore!

- Marcus Fedarko
- Gregor Seyer
- Nick Youngblut
- Antonio Garrido Fernandez
