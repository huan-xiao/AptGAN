import pandas as pd
import numpy as np
import random
import pickle
import tensorflow as tf
import torch
from feature_rna import RNA_Proteins_3Mer, RNA_Proteins_PseAAC, RNA_Proteins_SS, RNA_Aptamers_3Mer, RNA_Aptamers_PseKNC, RNA_Aptamers_SS, def_convs
from feature_dna import DNA_Proteins_3Mer, DNA_Proteins_PseAAC, DNA_Proteins_SS, DNA_Aptamers_3Mer, DNA_Aptamers_PseKNC, DNA_Aptamers_SS
from xgboost import XGBClassifier

noise_dim = 64
min_seq_len = 6 # changed from 6 to 20
max_seq_len = 185


# encode table
rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}


dna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3,
             "*":4}

rev_dna_vocab = {v:k for k,v in dna_vocab.items()}


def one_hot_encode(apt_type, seq, SEQ_LEN = max_seq_len):
    if apt_type=='RNA':
        mapping = dict(zip("ACGU*", range(5)))
    elif apt_type=='DNA':
        mapping = dict(zip("ACGT*", range(5)))
    else:
        print('aptamer type error!')
        return
         
    seq2 = [mapping[i] for i in seq]
    if len(seq2) < SEQ_LEN:
        extra = [np.eye(5)[4]] * (SEQ_LEN - len(seq2))
        return np.vstack([np.eye(5)[seq2] , extra])
    
    return np.eye(5)[seq2]


# to filter sequences with multiple segments
def seq_filter(apt_type, generated_sequences):
    matches = ["A", "C", "G", "U", "T"]
    seq_list = []
    
    for i in range(len(generated_sequences)):
        # transform number to vocab string
        seq = generated_sequences[i].numpy()
        seq_argmax = np.argmax(seq, 1)
        if apt_type == 'RNA':
            s = "".join(rev_rna_vocab[ind] for ind in seq_argmax)
        elif apt_type == 'DNA':
            s = "".join(rev_dna_vocab[ind] for ind in seq_argmax)
        
        # remove multiple segments sequences
        string_pre_str, str_ ,string_post_str = s.partition("*")
        if not any(x in string_post_str for x in matches):
            seq_list.append(s)
        
    return seq_list


# to clearn the * at the end of the sequences, , min_seq_len>=20
def seq_cleaner(generated_sequences, seq_min, seq_max):
    cleaned_seq = []
    
    for seq in generated_sequences:
        seq_pre, _, _ = seq.partition("*")
        if len(seq_pre)>=seq_min and len(seq_pre)<=seq_max:
            cleaned_seq.append(seq_pre)
    
    return cleaned_seq


def random_sample_vocab(apt_type, length):
    # encode list
    if apt_type == 'RNA':
        vocab_list = ["A", "C", "G", "U"]
    elif apt_type == 'DNA':
        vocab_list = ["A", "C", "G", "T"]
    
    return ''.join(random.choices(vocab_list, k=length))


# motif based sampling
def apt_sampling_from_motif(apt_type, seq_min, seq_max):
    # 1. choose a random motif
    # 2. choose whether to expand motif: True/False
    # 3. if False: break
    # 4. if True: choose a random num after the site_end
    # 5.        if No: break 
    # 6.        if Yes: repeat 2-4
    # 7. Random sampling the other regions, return
    
    if apt_type == 'RNA':
        motifs = pd.read_csv("./dataset/generating/RNA_motifs.csv")
    elif apt_type == 'DNA':
        motifs = pd.read_csv("./dataset/generating/DNA_motifs.csv")
    
    sample_ini = motifs[motifs["site_End"]<=seq_max].sample(1)
    site_sequence = sample_ini['site_Sequence'].item()
    site_start = sample_ini['site_Start'].item()
    site_end = sample_ini['site_End'].item()
    
    aptamer = random_sample_vocab(apt_type, site_start-1) + site_sequence
    
    while bool(random.getrandbits(1)):
        if site_end>=seq_max: # 157 is the largest site_Start
            break
        else:
            subsection_a = motifs[motifs["site_Start"]>site_end]
            subsection_b = motifs[motifs["site_End"]<=seq_max]
            intersection = pd.merge(subsection_a, subsection_b, on=['site_Sequence', 'site_Start', 'site_End'])
            
            if len(intersection)>0:
                sample = intersection.sample(1)
                site_sequence = sample['site_Sequence'].item()
                site_start = sample['site_Start'].item()
                aptamer += random_sample_vocab(apt_type, site_start-site_end-1)+site_sequence
                site_end = sample['site_End'].item()
    
    if len(aptamer)>seq_min:
        lower_bound = len(aptamer)
    else:
        lower_bound = seq_min
        
    cut_off = random.randint(lower_bound, seq_max) # the last position of vocab
    
    aptamer += random_sample_vocab(apt_type, cut_off-len(aptamer))+ "".join("*" for i in range(max_seq_len-cut_off))
    
    return aptamer


def featurize_apts_with_pro(apt_type, pro_file, pro_ss, apts):
    # read targeting protein
    pro_line_num=0
    pro = []
    
    with open(pro_file) as file:
        for line in file:
            pro_line_num+=1
            if pro_line_num==2: # only one protein
                line = line.strip()
                pro.append(line)
                break
    
    # load convs
    apt_file_num = len(apts)
    conv1, conv2, conv3 = def_convs()
    
    # featuring proteins and aptamers
    if apt_type == 'RNA':
        pro_3Mer = RNA_Proteins_3Mer(pro, conv1)
        pro_PseAAC = RNA_Proteins_PseAAC(pro)
        pro_SS = RNA_Proteins_SS(pro_ss, conv2)
    
        apts_3Mer = RNA_Aptamers_3Mer(apts, conv1)
        apts_PseKNC = RNA_Aptamers_PseKNC(apts)
        apts_SS = RNA_Aptamers_SS(apts, conv3)
    elif apt_type == 'DNA':
        pro_3Mer = DNA_Proteins_3Mer(pro, conv1)
        pro_PseAAC = DNA_Proteins_PseAAC(pro)
        pro_SS = DNA_Proteins_SS(pro_ss, conv2)
        
        apts_3Mer = DNA_Aptamers_3Mer(apts, conv1)
        apts_PseKNC = DNA_Aptamers_PseKNC(apts)
        apts_SS = DNA_Aptamers_SS(apts, conv3)
        
        
    pros_3Mer = pro_3Mer.repeat(apt_file_num, 1)
    pros_PseAAC = pro_PseAAC.repeat(apt_file_num, 1)
    pros_SS = pro_SS.repeat(apt_file_num, 1)
    
    # concatenate feature
    features_tensor = torch.cat((pros_3Mer, pros_PseAAC, pros_SS, apts_3Mer, apts_PseKNC, apts_SS), 1)

    # save feature tensors
    return features_tensor


def generate_aptamers(apt_type, seq_num, seq_min, seq_max, path):
    
    # import generator and discriminator
    if apt_type=='RNA':
        generator = tf.keras.models.load_model('./models/RNA/generator_epoch_9000.h5', compile=False)
        discriminator = tf.keras.models.load_model('./models/RNA/discriminator_epoch_9000.h5', compile=False)
    elif apt_type=='DNA':
        generator = tf.keras.models.load_model('./models/DNA/generator_epoch_4000.h5', compile=False)
        discriminator = tf.keras.models.load_model('./models/DNA/discriminator_epoch_4000.h5', compile=False)
    else:
        print("aptamer type error!")
        return
    
    
    # 1 generate sequences from trained generator
    seq_from_generator = []
    
    #print("\nGAN-based:")
    while len(seq_from_generator)<seq_num:
        #print(len(seq_from_generator), flush=True, end='\t')
        random_latent_vectors = tf.random.normal(shape=(seq_num, noise_dim))
        seq_temp = seq_cleaner(seq_filter(apt_type, generator(random_latent_vectors)), seq_min, seq_max)
        seq_from_generator.extend(seq_temp)
    
    seq_from_generator = seq_from_generator[:seq_num]
    
    #return seq_from_GAN
    file_name = path + 'seq_from_GAN.txt'
    with open(file_name, 'w') as fp:
        #pickle.dump(seq_from_generator, fp)
        for seq in seq_from_generator:
            fp.write(f"{seq}\n")
    
    if seq_max<4: # the length of the shortest aptamer motif is 4
        file_name = path + 'seq_gen.txt'
        with open(file_name, 'wb') as fp:
            pickle.dump(seq_from_generator, fp)
        return
    
    
    # 2 generate sequences from motif sampling
    seq_from_sampling = []
    
    #print("\nmotif-based:")
    while len(seq_from_sampling)<seq_num:
        #print(len(seq_from_sampling), flush=True, end='\t')
        seq_temp = [apt_sampling_from_motif(apt_type, seq_min, seq_max) for i in range(seq_num)]

        seq_onehot = np.asarray([one_hot_encode(apt_type, x) for x in seq_temp])
        seq_scores = discriminator(seq_onehot).numpy().squeeze()
        seq_indices = [i for i in (seq_scores <= 0).nonzero()[0]]

        for i in sorted(seq_indices, reverse=True):
            del seq_temp[i]

        seq_from_sampling.extend(seq_cleaner(seq_temp, seq_min, seq_max))
    
    seq_from_sampling = seq_from_sampling[:seq_num]
    
    #return seq_from_motif
    file_name = path + 'seq_from_motif.txt'
    with open(file_name, 'w') as fp:
        #pickle.dump(seq_from_sampling, fp)
        for seq in seq_from_sampling:
            fp.write(f"{seq}\n")
       
    
    # 3 combine the lists
    file_name = path + 'seq_gen.txt'
    seq_joint = seq_from_generator + seq_from_sampling
    with open(file_name, 'w') as fp:
        for seq in seq_joint:
            fp.write(f"{seq}\n")
    

def generate_aptamers_for_protein(apt_type, seq_num, seq_min, seq_max, threshold, path, pro_file, pro_ss): # this seq_num is for protein targeted, seq_max should be more than 4
    
    targeting_seq = []
    
    # import generator and discriminator
    if apt_type=='RNA':
        generator = tf.keras.models.load_model('./models/RNA/generator_epoch_9000.h5', compile=False)
        discriminator = tf.keras.models.load_model('./models/RNA/discriminator_epoch_9000.h5', compile=False)
    elif apt_type=='DNA':
        generator = tf.keras.models.load_model('./models/DNA/generator_epoch_4000.h5', compile=False)
        discriminator = tf.keras.models.load_model('./models/DNA/discriminator_epoch_4000.h5', compile=False)
    else:
        print("aptamer type error!")
        return
    
    
    while len(targeting_seq)<seq_num:
        
        # 1 generate sequences from trained generator
        seq_from_generator = []
        #print("GAN-based: ", end='')
        coe = max(50-seq_max+seq_min, 1)
        random_latent_vectors = tf.random.normal(shape=(seq_num*coe, noise_dim))
        seq_temp = seq_cleaner(seq_filter(apt_type, generator(random_latent_vectors)), seq_min, seq_max)
        seq_from_generator.extend(seq_temp)
        #print(len(seq_from_generator), flush=True, end='\t')
        
        # 2 generate sequences from motif sampling
        seq_from_sampling = []
        #print("motif-based: ", end='')
        seq_temp = [apt_sampling_from_motif(apt_type, seq_min, seq_max) for i in range(seq_num)]

        seq_onehot = np.asarray([one_hot_encode(apt_type, x) for x in seq_temp])
        seq_scores = discriminator(seq_onehot).numpy().squeeze()
        seq_indices = [i for i in (seq_scores <= 0).nonzero()[0]]

        for i in sorted(seq_indices, reverse=True):
            del seq_temp[i]

        seq_from_sampling.extend(seq_cleaner(seq_temp, seq_min, seq_max))
        #print(len(seq_from_sampling), flush=True, end='\t')
        
        # 3 combine the lists
        seq_joint = seq_from_generator + seq_from_sampling
        
        if len(seq_joint)==0:
            continue
            
        features_tensor = featurize_apts_with_pro(apt_type, pro_file, pro_ss, seq_joint)
        features_tensor = features_tensor.detach().numpy()
        
        # 4 judge the binding affinity
        xgb = XGBClassifier()
        if apt_type=='RNA':
            xgb.load_model("./models/RNA/XGB_classifier.json")
        elif apt_type=='DNA':
            xgb.load_model("./models/DNA/XGB_classifier.json")
        
        y_pred = xgb.predict_proba(features_tensor)[:,1]

        for i in range(len(y_pred)):
            if y_pred[i]>threshold:
                targeting_seq.append(seq_joint[i])
                
        #print("targeting seq: ", end='')       
        #print(len(targeting_seq), flush=True, end='\t')
        #print("")
        
    # 5 save to path
    targeting_seq = targeting_seq[:seq_num]
    file_name = path + 'targeting_seq.txt'
    with open(file_name, 'w') as fp:
        for seq in targeting_seq:
            fp.write(f"{seq}\n")
        
    