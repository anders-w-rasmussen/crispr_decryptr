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

<img align="left" src=/readme_files/figure_1.png alt="drawing" width="400"/>

There are three parts to the **CRISPR Decryptr** method:

First, the *infer* command takes gRNA counts as well as information about experimental design and returns guide-specific pertubation effects using a Markov Chain Monte Carlo inference procedure. 

Second, the *predict* command constructs a convolution matrix from intended targets. This command can align spacer sequences to a reference genome, or build the matrix from intended gRNA target locations. The convolution matrix serves as a linear map of guide-specific effect to base-specific effect.

Thirdly, the *classify* command takes the results of the first two commands to map the locations of functional elements on the region of interest using a Gaussian process deconvolution and Hidden semi-Markov Models. 

*Figure 1 from our publication CRISPR-Decryptr reveals  <br />  cis-regulatory elements from noncoding pertubation screens.*




# Get Started

After you've performed your CRISPR Screen, classifying regulatory elements just requires formatting your data and explaining your experimental design to the algorithm via some simple .tsv files. 

First let's make experiment_counts.tsv which contains our gRNA count data where each sample is a column, each gRNA sequence is a row. The first two lines (lets call our file counts.tsv) will look something like:

| gRNA ID | early_rep1  | early_rep2 | late_control_rep1  | late_control_rep2 | late_treat_rep1  | late_treat_rep2 |
| ----------- | ----------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| grna1 | 832  | 1002 | 764  | 10010 | 653  | 701 |
| grna2 | 878  | 943 | 832  | 1000 | 543  | 708 |

Our theoretical screen has an early timepoint and a late timepoint with both control and treatment conditions. We need to tell the algorithm this experimental design and what effects we're looking for with a .tsv file (lets call it dmat.tsv) in excel/ a text editor that looks like:

|  |  background_effect |  common_effect  | treatment_effect  | 
| ----------- | ----------- | ---------- | ---------- |
| early_rep1 | 1 | 0 | 0 |
| early_rep2 | 1 | 0 | 0 |
| late_control_rep1 | 1 | 1 | 0 |
| late_control_rep2 | 1 | 1 | 0 |
| late_treat_rep1 | 1 | 1 | 1 |
| late_treat_rep2 | 1 | 1 | 1 |

Here you can see the first column contains the sample names, the first row contains the "effect" names that we choose. The 1s and 0s in the file specify whether a certain effect of pertubation is considered as impacting that sample. For example, the background effect is present in all three samples across both time points (this could be thought of as the background propensity of sequencing a gRNA at all times). At the later timepoint, we have a "common" effect (which could be thought of as a treatment-independent impact of time on the cells) and treatment effect which only impacts the treated samples. If we want to understand the impact of pertubation in some treatment condition, it is important to account for any other effects that may be depleting or enriching our guides. 

Finally, we just need to make two a file containing intended target locations in the same guide order as the count .tsv. We can call this targets.tsv with the first two rows looking like

| chr10  |
| ---------- |
| 6027661 |
| 6027664 |

and a file rep_info.tsv which tells us which replicate each sample is from

|  | replicate|
| ---------- | ---------- |
| early_rep1 | 1 | 
| early_rep2 | 2 | 
| late_control_rep1 | 1 | 
| late_control_rep2 | 2 | 
| late_treat_rep1 | 1 | 
| late_treat_rep2 | 2 |

Now we can analyze our CRISPR screen!

```bash
decryptr infer counts.tsv dmat.tsv rep_info.tsv --outfile effects.tsv

decryptr predict targets.tsv False 

decryptr classify effects.tsv targets.tsv convolution_matrix.p 
```

The the last step will write files containing regulatory element calls for each of the effects under consideration, which can easily be opened in a genomic viewer. 

For more detailed information please see supplemental notes located here: 


# Credits

Co-First Authors on Paper: Anders Rasmussen, Tarmo Äijö  <br />
Method Development: Anders Rasmussen, Tarmo Äijö, Mariano Gabitto, Nick Carriero  <br />
Experimental Collaborators: Jane Skok, Neville Sanjana  <br />
PI: Richard Bonneau
