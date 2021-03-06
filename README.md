# K Nearest Neighbors in C with OpenMP

While this is intended to be a parallelized version of k nearest neighbors, it
unfortunately was initially hampered by my enthusiasm in making the CSV reader
somewhat automatic.

## Building

A basic `Makefile` is included free of charge, and has two targets `debug`,
which includes some erroneous printing for debugging purposes, as well as a
`knn` target, which is effectively a release target.

## Running

Help text shown below.

```
USAGE:
    %s [FLAGS and OPTIONS] FILE

Supplied data should be given in the same order/format as the input file, eg a
csv file with 2 real values, the label, then 2 more real values, a single data
point should be like so:

    real1,real2,real3,real4

ARGUMENTS:

    FILE    The name of a comma or tab separated value file, in which the first
            row can be the labels, which will be ignored. The specified file
            should not have more than one non-real field/column, which should
            be the label for that data entry. If any columns have a label, you
            must use one of the classification flags (-c or --classification).
            Otherwise, use the regression flags (-r or --regression). If you
            are classifying a file that does not have any labels, you must use
            the label option (-l or --label) to specify a column number to use
            as the label (0-indexed).

FLAGS:

    -c, --classification     Classify data read from stdin

    -r, --regression         Use a regression of the data in FILE to predict the
                             value of the dependent variable in the specified
                             column (-l or --label is required)

    -h, --help               Show this help text

OPTIONS:

    -l, --label-column       Column number to use as the label for regression; 
                             required when using the -r/--regression flag

    -k, --k-nearest          Number of nearest neighbors to use when
                             classifying or performing regression on an input
                             data point -- default is 5
```
