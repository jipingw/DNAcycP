DNAcycP Python package 
================

**Maintainer**: Ji-Ping Wang, \<<jzwang@northwestern.edu>\>; Keren Li, \<<keren.li@northwestern.edu>\>

**Licence**: GPLv3

**Cite DNAcycP package**:

Li, K., Carroll, M., Vafabakhsh, R., Wang, X.A. and Wang, J.-P., DNAcycP: A Deep Learning Tool for DNA Cyclizability Prediction, *Nucleic Acids Research*, 2021

## What is DNAcycP?

**DNAcycP**, short for **DNA** **cyc**lizablity **P**rediction, is a Python package for accurate predict of DNA intrinsic cyclizablity score. It was built upon a deep learning architecture with a hybrid of Inception and Residual network structure and an LSTM layer. DNAcycP was trained based on loop-seq data from Basu et al 2021 (see below). The predicted score, termed **C-score** achieves high accuracy compared to the experimentally measured cyclizablity score by loop-seq assay.

## Available format of DNAcycP

DANcycP is available in two formats: A web server available at http://DNAcycP.stats.northwestern.edu for real-time prediction and visualization of C-score up to 20K bp, and a standalone Python package available for free download from https://github.com/jipingw/DNAcycP. 


## Architecture of DNAcycP

The core of DNAcycP is a deep learning architecture mixed with an Inception-ResNet structure and an LSTM layer (IR+LSTM, Fig 1b) that processes the sequence and its reverse complement separately, the results from which are averaged and detrended to reach the predicted intrinsic score. (Fig 1a).

IR+LSTM starts with a convolutional layer for dimension reduction such that the encoded sequence space is reduced from 2D to 1D. The output is fed into an inception module that contains two parallel branches, each having two sequentially connected convolutional layers with branch-specific kernels to capture sequence features of different scale. The first branch has kernel dimension 3x1 for both layers and the second has kernel dimension 11x1 and 21x1 sequentially. The output of the inception module is combined by concatenation and added back to the input of the inception module to form a short circuit or residual network. Finally, the IR+LSTM concludes with a dense layer to predict output with linear activation function. 

![A diagram of DNAcycP.](./figures/Figure1.png)

## DNAcycP required packages

* `bio==1.3.3`
* `tensorflow==2.7.0`
* `keras==2.7.0`
* `pandas==1.3.5`
* `numpy==1.21.5`
* `docopt==0.6.2`
* `protobuf==3.20.0`


## Installation

**DNAcycP** Python package requires specific versions of dependencies. We recommend to install and run **DNAcycP** in a virtual environment. For example, suppose the downloaded DNAcycP package is unpacked as a folder `dnacycp-main`. We can install DNAcycP in a virtual environment as below:

```bash
cd DNAcycP-main
python3 -m venv env
source env/bin/activate test
pip3 install -e .
```

Run `dnacycp-cli ` to see whether it is installed properly.

```bash
dnacycp-cli 
```

Once done with DNAcycP for prediction, you can close the virtual environment by using:
```bash
deactivate
```

Once the virtual environment is deactivated, you need to re-activate it before you run another session of prediciotn as follows:
```bash
cd dnacycp-main
source env/bin/activate test
```

## Usage

DNAcycP supports the input sequence in two formats: FASTA format (with sequence name line beginning with “>”) or plain TXT format. Unlike in the web server version where only one sequence is allowed in input for prediction, the Python package allows multiple sequences in the same input file. In particular for the TXT format, each line (can be of different length) in the file is regarded as one input sequence for prediction. 

The main funciton in DNAcycP is `dancycp-cli`, which can be called as follows:
```bash
dnacycp-cli -f/-a/-t <inputfile> <basename>
or
dnacycp-cli -m <inputfile> <basename> <n>
```
where 
  * `-f/-a/-m/-t`: indicates the input file in FASTA or TXT format and the function we are using; one must be specified.
  * `<inputfile>`: is the name of the intput file;
  * `<basename>`: is the name base for the output file;
  * `<n>`: is the number of batchs you want to divide the chromosome sequence into.
  
### Example 1:

```bash
dnacycp-cli -f ./data/raw/ex1.fasta ./data/raw/ex1
```
The `-f` option specifies that the input file named "ex1.fasta" is in fasta format. 
The `./data/raw/ex1.fasta` is the sequence file path and name, and `./data/raw/ex1` specifies the output file will be saved in the directory `./data/raw` with file name initialized with `ex1`.

For example, `ex1.fasta` contains two sequences named ">seq1" and ">myseq2" respectively.

The output file will be named as "ex1_cycle_seq1.txt", "ex1_cycle_myseq2.txt"for the first and second sequences respectively. Each file contains three columns: `position`, `C_score_norm`, `C_score_unnorm`. The `C_score_norm` is the predicted C-score from the model trained based on the standardized loop-seq score of the tiling library of Basu et al 2021 (i.e. 0 mean unit variance). The `C_score_unnorm` is the predicted C-score recovered to the original scale of loop-seq score in the tiling library data from Basu et el 2021. The standardized loop-seq score provides two advantages. As loop-seq may be subject to a library-specific constant, standardized C-score is defined with a unified baseline as yeast genome (i.e. 0 mean in yeast genome). Secondly, the C-score provides statisitcal significance indicator, i.e. a C-score of 1.96 indicates 97.5% in the distribution.

### Example 2:

```bash
dnacycp-cli -a ./data/raw/ex1.fasta ./data/raw/ex1
```
The `-a` option means that the input file named "ex1.fasta" is in fasta format and the algorithm automaticly evenly divide the chromosomes into batches. The number of the batches is determined by how many times the chromosome length is 20,000,000. 

The `./data/raw/ex1.fasta` is the sequence file path and name, and `./data/raw/ex1` specifies the output file will be saved in the directory `./data/raw` with file name initialized with `ex1`.

If `ex1.fasta` contains two sequences named ">seq1" and ">myseq2" respectively and ">seq1" is less than 19,999,999 in length and ">myseq2" is 20,000,000 in length, then ">seq1" will be kept in one batch and ">myseq2" will be evenly divided into two batches. Note that this is not the case in `ex1.fasta`.

The output file will be named as "ex1_cycle_seq1_25.txt" for the first sequences and "ex1_cycle_myseq2_25.txt" and "ex1_cycle_myseq2_10000001.txt" for the second sequences. The number '25' and '10000001' marks the initial position of the batch.

### Example 3:

```bash
dnacycp-cli -m ./data/raw/ex1.fasta ./data/raw/ex1 n
```
The `-m` option means that the input file named "ex1.fasta" is in fasta format and n is the number of the batches the chromosomes are evenly divided into.

The output file will be named in similar way as in Example 2.


### Example 4:

```bash
dnacycp-cli -t ./data/raw/ex2.txt ./data/raw/ex2
```
With `-t` option, the input file is regarded as in TXT format, each line representing a sequence without sequence name line that begins with ">".
The predicted C-scores will be saved into two files, one with `_unnorm.txt` and the other with `_norm.txt` for unnormalized and normalized C-score, with C-scores in each line corresponding to the sequence in the input file in the same order.

For any input sequence, DNAcycP predicts the C-score for every 50 bp. Regardless of the input sequence format the first C-score in the output file corresponds to the sequence from position 1-50, second for 2-51 and so forth.

### Run prediction within Python interactive session

```python
from dnacycp import cycle_fasta, auto_cycle_fasta_threads, cycle_fasta_threads, cycle_txt
cycle_fasta("data/raw/ex1.fasta","example1")
auto_cycle_fasta_threads("data/raw/ex1.fasta","example1")
cycle_fasta_threads("data/raw/ex1.fasta","example1",n)
cycle_txt("data/raw/ex2.txt","example2")
```


## Other References

* Basu, A., Bobrovnikov, D.G., Qureshi, Z., Kayikcioglu, T., Ngo, T.T.M., Ranjan, A., Eustermann, S., Cieza, B., Morgan, M.T., Hejna, M. et al. (2021) Measuring DNA mechanics on the genome scale. Nature, 589, 462-467.


