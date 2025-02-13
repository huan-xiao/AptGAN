import argparse
from generation import generate_aptamers, generate_aptamers_for_protein


# Create a parser
parser = argparse.ArgumentParser(description="parameters")

# Add arguments
parser.add_argument('--type', type=int, help='generation type')
parser.add_argument('--seq_num', type=int, help='number of generated sequences')
parser.add_argument('--seq_min', type=int, help='minimum sequence length(>6)')
parser.add_argument('--seq_max', type=int, help='maximum sequence length (<186)')
parser.add_argument('--path', type=str, help='output path')
parser.add_argument('--threshold', type=float, help='threshold', default=0.6)
parser.add_argument('--pro_file', type=str, help='protein sequence in FASTA format', default=None)
parser.add_argument('--pro_ss', type=str, help='protein secondary structure', default=None)


# Parse the arguments
args = parser.parse_args()


if __name__ == "__main__":
    type = args.type
    seq_num = args.seq_num
    seq_min = args.seq_min
    seq_max = args.seq_max
    path = args.path
    threshold = args.threshold
    pro_file = args.pro_file
    pro_ss = args.pro_ss
    
    if type==0:
        generate_aptamers(seq_num, seq_min, seq_max, path)
    else:
        generate_aptamers_for_protein(seq_num, seq_min, seq_max, threshold, path, pro_file, pro_ss)
    
    
    