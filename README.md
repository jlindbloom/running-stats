# running-stats
A simple Python module for computing the sample statistics of an array in a running/online fashion (when samples are not kept).

This is just a Python port of [John Cook]'s C++ code [here](https://www.johndcook.com/blog/skewness_kurtosis/). There is also a Julia implementation by John Myles White [here](https://github.com/johnmyleswhite/StreamStats.jl). The only special feature of this code is that we support (coming soon) computing running statistics of arrays living on a GPU via CuPy.

You may also be interested in the Python module [RunStats](https://grantjenks.com/docs/runstats/).

