<p align="center">
<img src=/readme_files/decryptr_logo.png alt="drawing" width="550"/>
</p>

<p align="center">
Computational Method for the Analysis of CRISPR Noncoding Screens.   <br />
</p>

<p align="center">
Contact: Anders Rasmussen   <br />
email: arasmussen@flatironinstitute.org
</p>

<p align="center">
  
[Preprint on BiorXiv](https://www.biorxiv.org/content/10.1101/2020.08.13.247007v1 )
[Supplemental Notes (w/ Methods Section)](https://www.biorxiv.org/content/biorxiv/early/2020/08/14/2020.08.13.247007/DC1/embed/media-1.pdf?download=true)
  
 
</p>




# Installing

```bash
pip install git+https://git@github.com/anders-w-rasmussen/crispr_decryptr
```

# Requirements

We recommend running the method on hardware with at least 10GB of memory. As the method is highly parellelized, running on a multi-core CPU or cluster is ideal for large datasets. We tested the method on a MacBook Pro (Mojave) with 3.1 GHz Intel Core i7 + 16GB of RAM and were able to run published datasets within hours.

Please note that CRISPR-Decryptr requires Python 3. It also requires Xcode on Mac and wget to collect data off of the web (https://www.gnu.org/software/wget/). The code is intended for macOS and Linux.

# The Method 

<img align="right" src=/readme_files/figure_1.png alt="drawing" width="580"/>

There are three parts to the **CRISPR Decryptr** method:

First, the *infer* command takes gRNA counts as well as information about experimental design and returns guide-specific pertubation effects using a Markov Chain Monte Carlo inference procedure. 

Second, the *predict* command constructs a convolution matrix from intended targets. This command can align spacer sequences to a reference genome, or build the matrix from intended gRNA target locations. The convolution matrix serves as a linear map of guide-specific effect to base-specific effect.

Thirdly, the *classify* command takes the results of the first two commands to map the locations of functional elements on the region of interest using a Gaussian process deconvolution and Hidden semi-Markov Models. 


<br>


<br />

# Getting Started 

To get introduced to CRISPR-Decryptr, we can start by running the code on some simple simulated data. You don't need any experience in programming languages, just a terminal window and optionally Excel to get your data prepared in some simple .tsv files. 

Consider a theoretical screen for identifying regulatory elements that confer drug resistance. Our experimental design has two replicates and the following three conditions: an early condition, a control condition, and a treatment condition. 

### Step 1: Organize Data for Analysis

First, lets look at *grna_counts.tsv*. The first three rows of our raw gRNA count file will appear as follows:

| treatment_rep1 | control_rep1 | early_rep1 | treatment_rep2 | control_rep2 | early_rep2 |
|----------------|--------------|------------|----------------|--------------|------------|
| 43           | 60         | 121       | 145          | 144         | 150       |
| 84           | 80        | 126      | 157          | 193         | 182       |
| ...          | ...        | ...      | ...           | ...        | ...       |

Each column represents a different sample. The first row contains sample names, and the remaining rows represent gRNA counts corresponding to the targets in the target file *grna_targets.tsv*

| chr2 |
|----------------|
| 60034098          |
| 60034138         |
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

To learn about the algorithm in detail, please read **section 2.2** of the supplemental notes. In breif, the design matrix tells CRISPR-Decryptr which phenomena are impacting the gRNA counts in each condition. It is up to the user to specify a design matrix that best suits their analysis. 

Finally, we have the file *replicate_info.tsv* containing the replicate information for each sample as follows: 

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
CRISPR-Decryptr breaks apart the gRNA counts into batches. Smaller batch sizes allows the analysis to run faster at the expense of accuracy in its results. We suggest keeping this argument above 100. When this part of the method is done running (should take on the order of tens of minutes) it will produce the file *posterior_outfile.tsv*. 

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

If we open the folders titled "common" and "treatment" we can find the deconvolved signal, marginal state probabilities, and enhancer calls. If we open these files in a genomic viewer, and navigate to *chr2:60,038,379-60,046,564* it should look something like this


<p align="left">
<img src=/readme_files/demo.png alt="drawing" width="550"/>
</p>


Now that you've walked through all the different commands in CRISPR-Decryptr, we invite you to read the first two sections of our paper's supplemental notes to get familiar with all the arguments and funtionality of the method.


# Troubleshooting

First, please make sure you have the most recent version of CRISPR-Decryptr by running the following command:

```bash
pip install --upgrade git+https://git@github.com/anders-w-rasmussen/crispr_decryptr
```

### Troubleshooting the *infer* command

If you are having issues with Stan or the Stan model, we suggest manually installing *cmdstan*. The following commands will install *cmdstan* and then tell *cmdstanpy* where your install is:

```bash
git clone --recursive https://github.com/stan-dev/cmdstan
cd cmdstan
make build
export CMDSTAN='/path/to/cmdstan'
```

If you are using a cluster, make sure you load gcc:
```bash
module load gcc
```

If you see *TypeError: sample() got an unexpected keyword argument*, this might be because of an older versions of cmdstanpy which had different arguments. Please ensure you have the latest version of CRISPR-Decryptr installed and type:

```bash
pip install --upgrade cmdstanpy
```

### Troubleshooting the *predict* command

For the alignment algorithm, the *predict* command downloads genome references and uniqueness tracks off the internet into the species folder *"...python3.7/site-packages/decryptr/hg19"* (in the case of hg19). If CRISPR-Decryptr is downloading these files and something interrupts it there may be an error. Please delete the folder (in this case hg19) and try again. 

Hopefully it won't be required, but if you need to manually download these files for some reason, please try the following:

```bash
cd /where/folder/exists/...python3.7/site-packages/decryptr
mkdir hg19
cd hg19
mkdir fastas
mkdir uniqueness
```

Then please place these files in the fastas folder:
http://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/

And this file in the uniqueness folder:
http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDukeMapabilityUniqueness20bp.bigWig

There is an issue that might occur with TensorFlow 2.2.0. If you see an error from TensorFlow or Keras about the .h5 file, this is something that is going to be fixed in 2.3 (https://github.com/tensorflow/tensorflow/issues/38135). As such we will be including TensorFlow 2.0.0 as a dependency until 2.3 and suggest using CRISPR-Decrytr with this version. Reinstalling should take care of this, or you could pip install 2.0.0 directly with 

```bash
pip install tensorflow==2.0.0
```

### Troubleshooting the *classify* command

There is some memory-intensive matrix algebra *classify* step, so if you are doing a big CRISPR screen it is possible you may get a SIGKILL 9 error here. If this happens and you do not have access to a computer or cluster with more memory, you can divide your problem into smaller chunks that are managable on your hardware.  



# Credits

Co-First Authors on Paper: Anders Rasmussen, Tarmo Äijö  <br />
Method Development: Anders Rasmussen, Tarmo Äijö, Mariano Gabitto, Nick Carriero  <br />
Experimental Collaborators: Jane Skok, Neville Sanjana  <br />
PI: Richard Bonneau
