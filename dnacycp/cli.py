"""DNAcycP - DNA Sequence Cyclizability Prediction

Usage:
    dnacycp-cli -f <inputfile> <basename>
    dnacycp-cli -a <inputfile> <basename>
    dnacycp-cli -m <inputfile> <basename> <n>
    dnacycp-cli -t <inputfile> <basename>
    dnacycp-cli (-h | --help)
    
Arguments:
    <inputfile> Input file name.
    <basename>  Output file name base.
    <n> manually set threads number.
    
Options:
    -h --help   Show this screen.
    -f          FASTA mode.
    -a          auto threading for FASTA mode.
    -m          manual threading for FASTA mode.
    -t          TXT mode.
    
"""
from docopt import docopt
from dnacycp import cycle_fasta, cycle_txt, cycle_fasta_threads, auto_cycle_fasta_threads
import keras
import pandas as pd
import numpy as np
from numpy import array
from Bio import SeqIO

def main():
    arguments = docopt(__doc__)
    print("Input file: "+arguments['<inputfile>'])

    if arguments['-f']:
        cycle_fasta(arguments['<inputfile>'],
            arguments['<basename>'])
    if arguments['-a']:
        auto_cycle_fasta_threads(arguments['<inputfile>'],
            arguments['<basename>'])
    if arguments['-m']:
        cycle_fasta_threads(arguments['<inputfile>'],
            arguments['<basename>'], int(arguments['<n>']))
    if arguments['-t']:
        cycle_txt(arguments['<inputfile>'],
            arguments['<basename>'])

            
if __name__ == "__main__":
    main()
