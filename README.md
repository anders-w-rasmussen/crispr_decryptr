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

Please note that CRISPR-Decryptr requires Python 3.6. It also requires wget to collect data off of the web (https://www.gnu.org/software/wget/). The code is not intended for Window's Machines, and was written and tested on macOS (Mojave).


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

To get introduced to CRISPR-Decryptr, we can start by running the code on some simple simulated data. You don't need any experience in programming languages, just a terminal window and optionally Excel to get your data prepared in some simple .tsv files. 

Consider a theoretical screen for identifying regulatory elements that confer drug resistance. Our experimental design has two replicates and the following three conditions: an early condition, a control condition, and a treatment condition. 

### Step 1: Organize Data for Analysis

First, lets look at *grna_counts.tsv*. The first three rows of our raw gRNA count file will appear as follows:

| treatment_rep1 | control_rep1 | early_rep1 | treatment_rep2 | control_rep2 | early_rep2 |
|----------------|--------------|------------|----------------|--------------|------------|
| 1820           | 1231         | 1104       | 1923           | 1210         | 1166       |
| 2382           | 1356         | 1227       | 2321           | 1378         | 1219       |
| ...          | ...        | ...      | ...           | ...        | ...       |

Each column represents a different sample. The first row contains sample names, and the remaining rows represent gRNA counts corresponding to the targets in the target file *grna_targets.tsv*

| chr2 |
|----------------|
| 49068782          |
| 49067819         |
| ...          |

This file is just one column, it begins with the chromosome under consideration. Each successive entry is genomic position of the gRNA target. In this instance, the first guide targeting chr2:49068782 is in the second row of the file, which corresponds to the second row in the *gnra_counts.tsv* file. 

There are two more files we need to tell CRISPR-Decryptr about your experimental design. The first, *design_matrix.txt*, will tell the algorithm which "effects" impact which conditions. This file appears as follows, with the first column containing the same sample names, and the first row containing our "effect" names.

|  | early | control | treatment | 
|----------------|----------------|----------------|----------------|
| treatment_rep1          | 1         | 1      | 1           | 
| control_rep1           | 1         | 1       | 0           | 
| early_rep1           | 1        | 0       | 0          | 
| treatment_rep2           | 1        | 1       | 1           | 
| control_rep2        | 1       | 1      | 0           | 
| early_rep2         | 1        | 0      | 0          | 

To learn about the algorithm in detail, please read **section 2.2** of the supplemental notes. In breif, the design matrix tells CRISPR-Decryptr which phenomena are impacting the gRNA counts in each condition. It is up to the user to specify a design matrix that best suits their analysis. Finally, we have a file containing the replicate information for each sample as follows: 

|  | replicate | 
|----------------|----------------|
| treatment_rep1          | 1         | 
| control_rep1           | 1         | 
| early_rep1           | 1        | 
| treatment_rep2           | 2        |
| control_rep2        | 2     | 
| early_rep2         | 2        | 


### Step 2: Infer Guide-Specific Pertubation Effects (Infer)

Now that our files are all set, let's run CRISPR-Decryptr! We will start with the *infer* command, which will infer guide-specific regulatory effects from gRNA counts. In a terminal window, navigate to the directory where the files are we can begin.

To get an idea of what the infer command takes as arguments type:

```bash
decryptr infer -h
```
This will display the help message for the infer command. Take a look at **section 1** in the supplemental notes for additional information on each command and its arguments. You can aways type -h after any of the commands to see its help message. 

Let's use a relatively small batch size so this doesn't take too long. Type:

```bash
decryptr infer grna_counts.tsv design_matrix.tsv replicate_info.tsv --batch_size 100 --n_batches 4
```
CRISPR-Decryptr breaks apart the gRNA counts into batches. Smaller batch sizes allows the analysis to run faster at the expense of accuracy in its results. We suggest keeping this argument above 100. When this part of the method is done running (should take on the order of tens of minutes) it will produce the file *posterior_outfile.tsv*. We've included this so you don't need to wait for the analysis to complete. 

### Step 3: Create the Convolution Matrix (Predict)

Now we can run the *predict* command, which will create a convolution matrix which will be used in the next step to map the guide-specific effects from *posterior_outfile.tsv* to a base-by-base effect. 

```bash
decryptr predict grna_targets.tsv False
```

The second argument is False because we are not using the mutagenesis specific off-target scoring or repair outcome prediction (see **section 2.3**). This command will produce our convolution matrix *convolution_matrix.p*

### Step 3: Classify Regulatory Elements (Classify)

Finally, let's run the classify command to map the guide-specific effects from *posterior_outfile.tsv* to base-specific effects using the convolution matrix we just constructed. 


```bash
decryptr classify posterior_outfile.tsv grna_targets.tsv convolution_matrix.p
```

This will write folders containing the regulatory element calls and deconvolved effect tracks! 

### Step 4: Visualize the Results

If we open the folders titled "common" and "treatment" we can find the deconvolved signal and enhancer calls. If we open these files in a genomic viewer, and navigate to chr2:60,038,379-60,046,564 it should look something like this!


<p align="left">
<img src=/readme_files/demo.png alt="drawing" width="550"/>
</p>


Now that you've walked through all the different commands in CRISPR-Decryptr, we invite you to read the first two sections of our paper's supplemental notes to get familiar with all the arguments and funtionality of the method before analyzing your CRISPR screens. 



# Credits

Co-First Authors on Paper: Anders Rasmussen, Tarmo Äijö  <br />
Method Development: Anders Rasmussen, Tarmo Äijö, Mariano Gabitto, Nick Carriero  <br />
Experimental Collaborators: Jane Skok, Neville Sanjana  <br />
PI: Richard Bonneau
