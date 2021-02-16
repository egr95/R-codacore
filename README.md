# R-codacore

A self-contained, up-to-date implementation of [CoDaCoRe](https://www.biorxiv.org/content/10.1101/2021.02.11.430695v1), in the R programming language, by the original authors.

For an equivalent implementation in python, check [py-codacore](https://github.com/egr95/py-codacore). If you are interested in reproducing the results in the [original paper](add_arxiv_link), check [this repo](https://github.com/cunningham-lab/codacore).

Note this repository is under active development. If you would like to use CoDaCoRe on your dataset, and have any questions regarding the usage, implementation, or model itself, do not hesitate to contact <eg2912@columbia.edu>.

## How to run CoDaCoRe

1. To install codacore:

```r
devtools::install_github("egr95/R-codacore", ref="main")
```

2. To fit codacore on some data:
```r
library("codacore")
model = codacore(
    x, # compositional input, e.g., HTS count data 
    y, # response variable, typically a 0/1 binary indicator 
    type = "balances" # can use "amalgamations" instead, or abbreviations "B" and "A"
)
print(model)
plot(model)
```

3. Tensorflow

Note that codacore requires a working installation of tensorflow.
If you do not have tensorflow previously installed, when you run ```codacore()``` for the first time you will likely encounter an error message of the form:
```r
> codacore(x,y,type='B')

ERROR: Could not find a version that satisfies the requirement tensorflow
ERROR: No matching distribution found for tensorflow
Error: Installation of TensorFlow not found.

Python environments searched for 'tensorflow' package:
 /moto/stats/users/eg2912/miniconda3/envs/r-test/bin/python3.9
 /usr/bin/python2.7

You can install TensorFlow using the install_tensorflow() function.
```

This can be fixed simply by installing tensorflow, as follows:
```r
library("tensorflow")
install_tensorflow()
```

Note also that you may have to restart your R session between installation of codacore and/or tensorflow.

### Unsupervised learning

Coming soon... If you would like access to an early version, get [in touch](mailto:eg2912@columbia.edu).

### Multi-omics

Coming soon... If you would like access to an early version, get [in touch](mailto:eg2912@columbia.edu).
