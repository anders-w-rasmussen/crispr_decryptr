<p align="center">
<img src=/readme_files/decryptr_logo.png alt="drawing" width="550"/>
</p>

<p align="center">
Computational Method for the Analysis of CRISPR Noncoding Screens.   <br />
Preprint on BioRxiv:
</p>

<p align="center">
Contact: Anders Rasmussen   <br />
email: arasmussen@flatironinstitute.org
</p>




# Installing

```bash
pip install git+https://github.com/anders-w-rasmussen/crispr_decryptr
```

# The Method 

<img align="left" src=/readme_files/figure_1.png alt="drawing" width="600"/>

There are three parts to the **CRISPR Decryptr** method:

First, the *infer* command takes gRNA counts as well as information about experimental design and returns guide-specific pertubation effects using a Markov Chain Monte Carlo inference procedure. 

Second, the *predict* command constructs a convolution matrix from intended targets. This command can align spacer sequences to a reference genome, or build the matrix from intended gRNA target locations. The convolution matrix serves as a linear map of guide-specific effect to base-specific effect.

Thirdly, the *classify* command takes the results of the first two commands to map the locations of functional elements on the region of interest using a Gaussian process deconvolution and Hidden semi-Markov Models. 


<br>




<br />
<br />
<br />

<br />

# Getting Started 

To get introduced to CRISPR-Decryptr, we can start by running the code on some simple simulated data. This will also introduce the formats of the input files for each step in the method. 

Consider a theoretical screen for identifying regulatory elements that confer drug resistance. Our experimental design has two replicates and the following three conditions: an early condition, a control condition, and a treatment condition. Let's look at the files we need to create for analysis. 

First, lets look at grna_counts.tsv. The first three rows of our raw gRNA count file will appear as follows:

| treatment_rep1 | control_rep1 | early_rep1 | treatment_rep2 | control_rep2 | early_rep2 |
|----------------|--------------|------------|----------------|--------------|------------|
| 1820           | 1231         | 1104       | 1923           | 1210         | 1166       |
| 2382           | 1356         | 1227       | 2321           | 1378         | 1219       |
| ...          | ...        | ...      | ...           | ...        | ...       |

Each column represents a different sample. The first row contains sample names, and the remaining rows represent gRNA counts corresponding to the targets in the target file grna_targets.tsv

| chr2 |
|----------------|
| 49068782          |
| 49067819         |
| ...          |

This file is just one column, it begins with the chromosome under consideration. Each successive entry is genomic position of the gRNA target. In this instance, the first guide targeting chr2:49068782 is in the second row of the file, which corresponds to the second row in the gnra_counts.tsv file. 










# Credits

Co-First Authors on Paper: Anders Rasmussen, Tarmo Äijö  <br />
Method Development: Anders Rasmussen, Tarmo Äijö, Mariano Gabitto, Nick Carriero  <br />
Experimental Collaborators: Jane Skok, Neville Sanjana  <br />
PI: Richard Bonneau
