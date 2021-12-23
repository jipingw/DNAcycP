# DNAcycP: A Deep Learning Tool for DNA Cyclizability Prediction 

## DNA bendability and DNAcycP

DNA bendability is one fundamental mechanic property that affects various cellular functions. Recently a high throughput assay named loop-seq was developed to allow measuring of intrinsic cyclizability of massive DNA fragments simultaneously (Basu et al 2021). Using the loop-seq data, we develop a software tool, DNAcycP, based on a deep-learning approach for intrinsic DNA cyclizability prediction. We develop a computational tool named DNAcycP based on a deep learning model. DNAcycP prediction is solely based on DNA sequence and it features simplicity and efficiency in usage. 

DNAcycP predicts intrinsic DNA cyclizability with high fidelity compared to loop-seq data. Using an independent dataset from an in vitro study of DNA looping propensity (Rosanio et al 2015) we further verified the predicted cyclizability score, termed C-score, can well distinguish DNA fragments with different looping propensity. We applied DNAcycP to multiple species ranging from bacteria to mammals, and found predicted DNA cyclizability scores differ in magnitude between species, while preserving the same patterns at nucleosome dyad-proximal regions. Additionally, DNAcycP reveals intriguing cyclizability patterns at transcription factor binding sites. In particular at CTCF site, the cyclizability is substantially elevated and well preserved across species.

## Reference
If you use this code, please cite the following [paper]()

    @inproceedings{dnacycp2021,
       author = {Li, K. and Carroll, M. and Vafabakhsh, R. and Wang, X.A. and Wang, J.P.},
        title = "{DNAcycP: A Deep Learning Tool for DNA Cyclizability Prediction}",
        year = 2021,
    }


* Basu, A., Bobrovnikov, D.G., Qureshi, Z., Kayikcioglu, T., Ngo, T.T.M., Ranjan, A., Eustermann, S., Cieza, B., Morgan, M.T., Hejna, M. et al. (2021) Measuring DNA mechanics on the genome scale. Nature, 589, 462-467.
* Rosanio, G., Widom, J. and Uhlenbeck, O.C. (2015) In vitro selection of DNAs with an increased propensity to form small circles. Biopolymers, 103, 303-320.
* Brogaard, K., Xi, L., Wang, J.P. and Widom, J. (2012) A map of nucleosome positions in yeast at base-pair resolution. Nature, 486, 496-501.

## Architecture of DNAcycP

DNAcycP takes the one-hot encoding of every 50 bp DNA sequence and its reverse complement as input. The core of DNAcycP is a deep learning architecture pipeline mixed with Inception-ResNet structure and an LSTM layer (IR+LSTM, Fig 1b) that processes the sequence and its reverse complement separately, the results from which are averaged and detrended to reach the predicted intrinsic score. (Fig 1a)

IR+LSTM starts with a convolutional layer for dimension reduction such that the encoded sequence space is reduced from 2D to 1D. The output is fed into an inception module that contains two parallel branches, each having two sequentially connected convolutional layers with branch-specific kernels to capture sequence features of different scale. The first branch has kernel dimension 3x1 for both layers and the second has kernel dimension 11x1 and 21x1 sequentially. The output of the inception module is combined by concatenation and added back to the input of the inception module to form a short circuit or residual network. After a 2x1 max-pooling, a batch-normalization and a dropout layers, the resulting layer is then passed onto an LSTM layer with 20 hidden memory units, followed by a dropout layer with a ratio of 0.2. Finally, the IR+LSTM concludes with a dense layer to predict output with linear activation function. The ReLU function was applied to the output of each convolution layers, followed subsequently by 2x1 max-pooling (except first convolutional layer on each branch), batch-normalization and a dropout with a ratio of 0.2. The stride was equal to 1 throughout. The different kernel dimensions in the inception branches were chosen for the consideration to capture sequence properties including codon, poly-AT tracks or 10-bp periodicity of dinucleotide motifs that have been shown in the literature to affect DNA sequence flexibility (see Results). The LSTM layer further provides a holistic capturing of sequencing information in the scale of entire input sequence length such as the strength of periodicity of key dinucleotide motifs and their phase angles etc.

![A diagram of DNAcycP.](Figure1.png){width=80%} 

## Relevant experimental data sets

We considered five loop-seq data sets from Basu et al (2021) for model training and comparisons,  including: 1. S. cerevisiae nucleosome library of 19,907 different sequences of 50 bp selected from immediate upstream or downstream of the dyads (dyads not included) of 10,000 nucleosomes with highest NCP scores from Brogaard et al (2012) in S. cerevisiae SacCer2 genome; 2. random sequence library of 12,472 sequences generated with equal expected frequency of A/C/GT; 3. a tiling library 82,368 sequences from 576 genes that were selected from yeast genome whose ORFs ends were both mapped with high confidence, among which the first 297 were randomly chosen and the subsequent 279 had highest expression values. For each gene, the +1 nucleosome dyad position +/- 2,000 bp region was first selected, and 50 bp sequences within this region were extracted with tiling spacing of 7 bp; and 4. yeast ChrV library of 82,404 50 bp sequences tiled with 7bp spacing.

The fifth data set was from an independent in vitro study for DNA propensity for looping (Rosanio et al 2015). The initial library L0 contained ~2.4x10^15 species of 90 bp DNA fragments synthesized randomly. The subsequent libraries L1, L2, …, L6 contained 90 bp DNA fragments that successfully formed loops in previous rounds under different experimental conditions. For L0 - L3, the reaction volume remained constant as 2.18 l, while the ligation time monotonically decreased from 30 min, 15 min, 10 min to 4 min sequentially. For L4, L5 the reaction volume was reduced to 500 ml and the ligation time decreased to 1 min and 10 s respectively. We randomly selected 100,000 fragments from each library to evaluate the prediction of DNAcycP.

Experimented various deep learning architectures on different data sets, the final IR+LSTM model is trained by the tiling library.

## Usage

DNAcycP supports two modes of input: FASTA format (with sequence name lines beginning with “>”) and plain TEXT format.

To call FASTA format mode, specify input file through  `-i`, and base name (path) of output through `-b`. For each sequence of length n>=50 bp, DNAcycP predicts c-score of every 50-bp DNA fragment in sequence, i.e., the first output score corresponds to the DNA sub-sequence from position 1 to 50 and so forth. Output of each sequence is a text file with columns `position` and `c-score`, stored in an individual file.

```bash
python3 dnasysp.py -i ex1.fasta -b ex1
```

To call TEXT format mode, just simply add argument  `-t` in the command line. DNAcycP treat each line as an individual sequence and output a single text file with multiple line of c-scores, where each line corresponses to an input sequence. 

```bash
python3 dnasysp.py -i ex2.txt -b ex2 -t
```


