# R-codacore

A self-contained, up-to-date implementation of [CoDaCoRe](https://www.biorxiv.org/content/10.1101/2021.02.11.430695v1), in the R programming language, by the original authors.

For an equivalent implementation in python, check [py-codacore](https://github.com/egr95/py-codacore). If you are interested in reproducing the results in the [original paper](add_arxiv_link), check [this repo](https://github.com/cunningham-lab/codacore).

Note this repository is under active development. If you would like to use CoDaCoRe on your dataset, and have any questions regarding the installation, usage, implementation, or model itself, do not hesitate to contact <eg2912@columbia.edu>.
Contributions and fixes are also welcome - please create an issue, submit a pull request, or email me.

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

3. Tensorflow and Keras:

Note that CoDaCoRe requires a working installation of Tensorflow.
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

Note also that you may have to restart your R session between installation of codacore, tensorflow, and keras.

4. Running on an HPC cluster:

Again, you must have a working installation of Tensorflow and Keras.
Depending on your permissions, you may have to install these to a personal directory, using ```.libPaths```.
My personal recommendation is to skip ```.libPaths``` altogether and use an R conda environment, e.g.,
```sh
conda create -n <my-env-name> r-essentials r-base
conda activate <my-env-name>
conda install -c conda-forge r-keras
# Whenever possible, it is more robust to install dependencies via conda than install.packages
conda install -c conda-forge r-devtools 
```

### Unsupervised learning

Coming soon... If you would like access to an early version, get [in touch](mailto:eg2912@columbia.edu).

### Multi-omics

Coming soon... If you would like access to an early version, get [in touch](mailto:eg2912@columbia.edu).
