<img src=/readme_files/decryptr_logo.png alt="drawing" width="350"/>

Computational Method for the Analysis of CRISPR Noncoding Screens

# Installing

```bash
pip install git+https://github.com/anders-w-rasmussen/crispr_decryptr
```

# The Method 

<img align="left" src=/readme_files/figure_1.png alt="drawing" width="500"/>

There are three parts to the **CRISPR Decryptr** method:

First, the *infer* command takes gRNA counts as well as information about experimental design and returns guide-specific pertubation effects using a Markov Chain Monte Carlo inference procedure. 

Second, the *predict* command constructs a convolution matrix from intended targets. This command can align spacer sequences to a reference genome, or build the matrix from intended gRNA target locations. The convolution matrix serves as a linear map of guide-specific effect to base-specific effect.

Thirdly, the *classify* command takes the results of the first two commands to map the locations of functional elements on the region of interest using a Gaussian process deconvolution and Hidden semi-Markov Models. 

```bash
decryptr infer <count_filename> <design_matrix_filename> <replicate_information_filename> [-h] [options]

decryptr predict <targets> <cas9_alg> [-h] [options]

decryptr classify <effect_file> <targets> <conv_mat> [-h] [options]
```
