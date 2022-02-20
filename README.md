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


## Installation

**DNAcycP** Python package requires specific versions of dependencies. We recommend to install and run **DNAcycP** in a virtual environment. For example, suppose the downloaded DNAcycP package is unpacked as a folder `dnacycp-main`. We can install DNAcycP in a virtual environment as below:

```bash
cd dnacycp-main
python3 -m venv env
source env/bin/activate test
pip install -e .
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
dnacycp-cli -f/-t <inputfile> <basename>
```
where 
  * `-f/-t`: indicates the input file name in FASTA or TXT format respectively;
  * `<inputfile>`: is the name of the intput file;
  * `<basename>`: is the name base for the output file.

Example 1:

```bash
dnacycp-cli -f ./data/raw/ex1.fasta ./data/raw/ex1
```
The `-f` option specifies that the input file named "ex1.fasta" is in fasta format. 
Without `-t` option, the input file is regarded as in FASTA format. The predicted C-score will be saved in files, one for each sequence with two columns named `Position` and `C-score`. 
The `./data/raw/ex1.fasta` is the sequence file path and name, and `./data/raw/ex1` specifies the output file will be saved in the directory `./data/raw` with file name marked as `ex1`.
For example, `ex1.fasta` contains two sequences named ">seq1" and ">myseq2" respectively.
The output file will be named as "ex1_cycle_seq1.txt", "ex1_cycle_myseq2.txt"for the first and second sequences respectively.


Example 2:

```bash
dnacycp-cli -t ./data/raw/ex2.txt ./data/raw/ex2
```
With `-t` option, the input file is regarded as in TXT format, each line representing a sequence without sequence name line that begins with ">".
The predicted C-scores will be saved in one file, with C-scores in each line corresponding to the sequence in the input file in the same order.

For any input sequence, DNAcycP predicts the C-score for every 50 bp. Regardless of the input sequence format the first C-score in the output file corresponds to the sequence from position 1-50, second for 2-51 and so forth.


## Other References

* Basu, A., Bobrovnikov, D.G., Qureshi, Z., Kayikcioglu, T., Ngo, T.T.M., Ranjan, A., Eustermann, S., Cieza, B., Morgan, M.T., Hejna, M. et al. (2021) Measuring DNA mechanics on the genome scale. Nature, 589, 462-467.


