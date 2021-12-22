# R-codacore

A self-contained, up-to-date implementation of [CoDaCoRe](https://doi.org/10.1093/bioinformatics/btab645), in the R programming language, by the original authors.

The [CoDaCoRe guide](https://egr95.github.io/R-codacore/inst/misc/guide.html) contains a detailed tutorial on usage and functionality (note this tutorial assumes a prior installation of the package as per the steps below).

Note this repository is under active development. If you would like to use CoDaCoRe on your dataset, and have any questions regarding the installation, usage, implementation, or model itself, do not hesitate to contact <eg2912@columbia.edu>. Some previously asked questions are available on the [Issues page](https://github.com/egr95/R-codacore/issues).
Contributions, fixes, and feature requests are also welcome - please create an issue, submit a pull request, or email me.

## How to install and run CoDaCoRe

1. In order to install codacore from GitHub, we will need the [devtools package](https://www.r-project.org/nosvn/pandoc/devtools.html).

```r
devtools::install_github("egr95/R-codacore", ref="main")
```

2. To fit codacore on some data:
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

3. Tensorflow and Keras:

Note that CoDaCoRe requires a working installation of [TensorFlow](https://tensorflow.rstudio.com/).
If you do not have Tensorflow previously installed, when you run ```codacore()``` for the first time you will likely encounter an error message of the form:
```r
> codacore(x, y)

ERROR: Could not find a version that satisfies the requirement tensorflow
ERROR: No matching distribution found for tensorflow
Error: Installation of TensorFlow not found.

Python environments searched for 'tensorflow' package:
 /moto/stats/users/eg2912/miniconda3/envs/r-test/bin/python3.9
 /usr/bin/python2.7

You can install TensorFlow using the install_tensorflow() function.
```

This can be fixed simply by [installing tensorflow](https://tensorflow.rstudio.com/installation/), as follows:
```r
install.packages("tensorflow")
library("tensorflow")
install_tensorflow()

install.packages("keras")
library("keras")
install_keras()
```

Note also that you may have to restart your R session between installation of ```codacore```, ```tensorflow```, and ```keras```.

4. Running on an HPC cluster:

Again, you must have a working installation of Tensorflow and Keras.
Depending on your permissions, you may have to install these to a personal directory, using ```.libPaths```.
My personal recommendation is to skip ```.libPaths``` altogether and use a conda environment, e.g.,
```sh
conda create -n <my-env-name> r-essentials r-base
conda activate <my-env-name>
conda install -c conda-forge r-keras
# Whenever possible, it is more robust to install dependencies via conda than install.packages
conda install -c conda-forge r-devtools 
```

## Additional functionality

Some of the additional functionality of our package, including unsupervised learning and multi-omics, is discussed in the [Guide](https://egr95.github.io/R-codacore/inst/misc/guide.html). For feature requests, or to get access to an early version, get [in touch](mailto:eg2912@columbia.edu).


## Acknowledgements
Thanks for your contributions to R-codacore!

- Marcus Fedarko
