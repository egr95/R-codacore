# R-codacore

A self-contained, up-to-date implementation of [CoDaCoRe](add_arxiv_link), in the R programming language, by the original authors.

For an equivalent implementation in python, check [py-codacore](https://github.com/egr95/py-codacore). If you are interested in reproducing the results in the [original paper](add_arxiv_link), check [this repo](https://github.com/cunningham-lab/codacore).

Note this repository is under active development. If you would like to use CoDaCoRe on your dataset, and have any questions at all regarding the usage, implementation, or model itself, do not hesitate to contact <eg2912@columbia.edu>.

## How to run CoDaCoRe

1. To install codacore:
    devtools::install.packages()

2. To fit codacore on some data:
    library("codacore")
    model = codacore(x, y)
    print(model)
    plot(model)
