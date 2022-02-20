import keras
import pandas as pd
import numpy as np
from numpy import array
from Bio import SeqIO

network_final = keras.models.load_model("irlstm")
detrend_int = 0.018608475103974342
detrend_slope = 1.033974289894104

def dnaOneHot(sequence):
    seq_array = array(list(sequence))
    code = {"A": [0], "C": [1], "G": [2], "T": [3], "N": [4],
            "a": [0], "c": [1], "g": [2], "t": [3], "n": [4]}
    onehot_encoded_seq = []
    for char in seq_array:
        onehot_encoded = np.zeros(5)
        onehot_encoded[code[char]] = 1
        onehot_encoded_seq.append(onehot_encoded[0:4])
    return onehot_encoded_seq

def cycle_fasta(inputfile, outputbase):
    genome_file = SeqIO.parse(open(inputfile),'fasta')
    for fasta in genome_file:
        chrom = fasta.id
        genome_sequence = str(fasta.seq)
        onehot_sequence = dnaOneHot(genome_sequence)
        onehot_sequence = array(onehot_sequence)
        onehot_sequence = onehot_sequence.reshape((onehot_sequence.shape[0],4,1))
        print("sequence length: "+chrom+" "+str(onehot_sequence.shape[0]))
        fit = []
        fit_reverse = []
        for ind_local in np.array_split(range(25, onehot_sequence.shape[0]-24), 100):
            onehot_sequence_local = []
            for i in ind_local:
                s = onehot_sequence[(i-25):(i+25),]
                onehot_sequence_local.append(s)
            onehot_sequence_local = array(onehot_sequence_local)
            onehot_sequence_local = onehot_sequence_local.reshape((onehot_sequence_local.shape[0],50,4,1))
            onehot_sequence_local_reverse = np.flip(onehot_sequence_local,[1,2])
            fit_local = network_final.predict(onehot_sequence_local)
            fit_local = fit_local.reshape((fit_local.shape[0]))
            fit.append(fit_local)
            fit_local_reverse = network_final.predict(onehot_sequence_local_reverse)
            fit_local_reverse = fit_local_reverse.reshape((fit_local_reverse.shape[0]))
            fit_reverse.append(fit_local_reverse)
        fit = [item for sublist in fit for item in sublist]
        fit = array(fit)
        fit_reverse = [item for sublist in fit_reverse for item in sublist]
        fit_reverse = array(fit_reverse)
        fit = detrend_int + (fit + fit_reverse) * detrend_slope/2
        n = fit.shape[0]
        fitall = np.vstack((range(25,25+n),fit))
        fitall = pd.DataFrame([*zip(*fitall)])
        fitall.columns = ["posision","c_score"]
        fitall = fitall.astype({"posision": int})
        fitall.to_csv(outputbase+"_cycle_"+chrom+".txt", index = False)
        print("Output file: "+outputbase+"_cycle_"+chrom+".txt")

def cycle_txt(inputfile, outputbase):
    with open(inputfile) as f:
            input_sequence = f.readlines()
    output_cycle = []
    for loop_sequence in input_sequence:
        loop_sequence =loop_sequence.rstrip()
        l = len(loop_sequence)
        onehot_loop = dnaOneHot(loop_sequence)
        onehot_loop = array(onehot_loop)
        onehot_loop = onehot_loop.reshape((l,4,1))
        onehot_loops = []
        for i in range(l-49):
            onehot_loops.append(onehot_loop[i:i+50])
        onehot_loops = array(onehot_loops)
        onehot_loops_reverse = np.flip(onehot_loops,[1,2])
        cycle_local = network_final.predict(onehot_loops)
        cycle_local_reverse = network_final.predict(onehot_loops_reverse)
        cycle_local = detrend_int + (cycle_local + cycle_local_reverse) * detrend_slope/2
        cycle_local = cycle_local.reshape(cycle_local.shape[0])
        output_cycle.append(cycle_local)
    with open(outputbase+"_cycle.txt", "w") as file:
        for row in output_cycle:
            s = " ".join(map(str, row))
            file.write(s+'\n')
