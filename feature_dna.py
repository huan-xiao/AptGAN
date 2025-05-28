import pandas as pd
import numpy as np
import propy
from propy.PseudoAAC import GetAPseudoAAC
from repDNA.psenac import PseKNC
import RNA
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle


aa_int = {'A':1,'R':2,'N':3,'D':4,'C':5,'E':6,'Q':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20} 

nt_int = {'A':1,'C':2,'G':3,'T':4}


def Proteins_3Mer(data, conv1):
    proteins_kmer = [] # 1-mer encoding
    max_pro_len = 3999 # requirement for input

    for pro in data:
        pro_seq = []

        for aa in pro:
            pro_seq.append(aa_int[aa])

        pro_seq = np.pad(pro_seq, (0, max_pro_len-len(pro_seq)), 'constant', constant_values=0)
        proteins_kmer.append(pro_seq)
        
    proteins_kmer = np.array(proteins_kmer)
    protein_input = torch.tensor(proteins_kmer, dtype=torch.float32)
    protein_input = protein_input.reshape(-1, 1, 3999)
    protein_output = conv1(protein_input)
    
    protein_output = protein_output.reshape(-1, 3997)
    return  protein_output


    
def Proteins_PseAAC(data):
    proteins_PseAAC = []

    for pro in data:
        PseAAC = list(GetAPseudoAAC(pro, lamda=5).values())
        proteins_PseAAC.append(np.array(PseAAC))
        #lamda should NOT be larger than the length of input protein sequence, the shortest is 6, so reset to 5
    
    proteins_PseAAC = np.array(proteins_PseAAC)
    Proteins_PseAAC = torch.tensor(proteins_PseAAC, dtype=torch.float32)
    return Proteins_PseAAC
    

    
def Proteins_SS(filename, conv2):
    HEC = []
    max_HEC_len = 399
    file_line_num=0
    
    with open(filename) as file:
        for line in file:
            file_line_num+=1
            if file_line_num%3==0:
                line = line.strip()
                line_len = len(line)
                
                H = []
                E = []
                C = []

                # measure each SS
                current_a = line[0]
                num = 0

                for alphabet in line:
                    if alphabet!=current_a:
                        # record the length ratio of last alphabet
                        if current_a=='H':
                            H.append(num/line_len)
                            E.append(0.)
                            C.append(0.)
                        elif current_a=='E':
                            H.append(0.)
                            E.append(num/line_len)
                            C.append(0.)
                        elif current_a=='C':
                            H.append(0.)
                            E.append(0.)
                            C.append(num/line_len)

                        current_a = alphabet
                        num = 1
                    else:
                        num += 1

                # for the last alphabet, which can only be recorded afterwards
                if current_a=='H':
                            H.append(num/line_len)
                            E.append(0.)
                            C.append(0.)
                elif current_a=='E':
                    H.append(0.)
                    E.append(num/line_len)
                    C.append(0.)
                elif current_a=='C':
                    H.append(0.)
                    E.append(0.)
                    C.append(num/line_len)

                
                # concatenate H, E, C to HEC
                if max_HEC_len>len(H):
                    H_pad = np.pad(H, (0, max_HEC_len-len(H)), 'constant', constant_values=0).reshape(1,-1)
                else:
                    H_pad = np.array(H[:max_HEC_len]).reshape(1,-1) # truncated if exceed max_HEC_len
                
                if max_HEC_len>len(E):
                    E_pad = np.pad(E, (0, max_HEC_len-len(E)), 'constant', constant_values=0).reshape(1,-1)
                else:
                    E_pad = np.array(E[:max_HEC_len]).reshape(1,-1)
                
                if max_HEC_len>len(C):
                    C_pad = np.pad(C, (0, max_HEC_len-len(C)), 'constant', constant_values=0).reshape(1,-1)
                else:
                    C_pad = np.array(C[:max_HEC_len]).reshape(1,-1)

                temp = np.concatenate((H_pad, E_pad, C_pad), axis=0)
                HEC.append(temp)

    file.close()
    HEC = np.array(HEC)
    HEC_input = torch.tensor(HEC, dtype=torch.float32)
    HEC_output = conv2(HEC_input)
    
    HEC_output = HEC_output.reshape(-1, 398)
    return HEC_output
    
    
    
def DNA_Aptamers_3Mer(data, conv1):
    aptamers_kmer = []
    max_apt_len = 185

    for apt in data:
        apt_seq = []
        for nt in apt:
            apt_seq.append(nt_int[nt])

        apt_seq = np.pad(apt_seq, (0, max_apt_len-len(apt_seq)), 'constant', constant_values=0)
        aptamers_kmer.append(apt_seq)
        
    aptamers_kmer = np.array(aptamers_kmer)
    aptamers_input = torch.tensor(aptamers_kmer, dtype=torch.float32).reshape(-1, 1, 185)
    aptamers_out = conv1(aptamers_input)
    
    aptamers_out = aptamers_out.reshape(-1, 183)
    return aptamers_out

    
    
def DNA_Aptamers_PseKNC(data):
    pseknc = PseKNC()

    aptamers_PseKNC = []

    for apt in data:
        apt = apt.replace('U','T')
        vec = pseknc.make_pseknc_vec([apt])
        aptamers_PseKNC.append(np.array(vec).flatten())
    
    aptamers_PseKNC = np.array(aptamers_PseKNC)
    Aptamers_PseKNC = torch.tensor(aptamers_PseKNC, dtype=torch.float32)
    return Aptamers_PseKNC
    


def DNA_Aptamers_SS(data, conv3):# data, conv3
    
    aptamers_fold = []
    
    for apt in data:
        fc  = RNA.fold_compound(apt)
        (ss, _) = fc.mfe()
        aptamers_fold.append(ss)
        
    UP = []
    max_UP_len = 93 # 185/2
    for line in aptamers_fold:
            line = line.strip()
            line_len = len(line)
            U = [] # un pair
            P = [] # pair

            # measure each SS
            current_a = line[0]
            num = 0
            
            for alphabet in line:
                if alphabet!=current_a:
                    # record the length ratio of last alphabet
                    if current_a=='.':
                        U.append(num/line_len)
                        P.append(0.)
                    else:
                        U.append(0.)
                        P.append(num/line_len)

                    current_a = alphabet
                    num = 1
                else:
                    num += 1

            # for the last alphabet, which can only be recorded afterwards
            if current_a=='.':
                U.append(num/line_len)
                P.append(0.)
            else:
                U.append(0.)
                P.append(num/line_len)

            # concatenate U, P to UP
            U_pad = np.pad(U, (0, max_UP_len-len(U)), 'constant', constant_values=0).reshape(1,-1)
            P_pad = np.pad(P, (0, max_UP_len-len(P)), 'constant', constant_values=0).reshape(1,-1)

            temp = np.concatenate((U_pad, P_pad), axis=0)
            UP.append(temp)
            
    UP = np.array(UP)
    UP_input = torch.tensor(UP, dtype=torch.float32)
    UP_output = conv3(UP_input)
    
    UP_output = UP_output.reshape(-1, 92)
    return UP_output
    
    
    
def def_convs():
    # custom CNN 1D weights, this CNN is used for Kmer of protein and aptamer
    conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3, bias=False)
    K1 = torch.tensor([[[0.1, 0.456, 0.99]]])
    conv1.weight = nn.Parameter(K1)
    
    # custom CNN 1D weights, this CNN is used for protein secondary structure
    conv2 = nn.Conv1d(in_channels=3,out_channels=1,kernel_size=2, bias=False)
    K2 = torch.tensor([[[1.1, 2.1],[1.456, 2.456],[1.99, 2.99]]]) 
    conv2.weight = nn.Parameter(K2)
    
    # custom CNN 1D weights, this CNN is used for aptamer secondary structure
    conv3 = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=2, bias=False)
    K3 = torch.tensor([[[1.1, 2.1],[1.456, 2.456]]])
    conv3.weight = nn.Parameter(K3)
    
    return conv1, conv2, conv3
    
    
    
if __name__ == "__main__":
    
    data = pd.read_csv('./dataset/training_dna/data.csv')
    
    pros = data['Protein']
    apts = data['Aptamer']
    pros_ss = "./dataset/training_dna/Proteins_SS.fas"
    
    conv1, conv2, conv3 = def_convs()
    
    # Protein 3Mer 
    pros_3Mer = Proteins_3Mer(pros, conv1)
    torch.save(pros_3Mer, './dataset/training_dna/Proteins_3Mer.pt')
    
    # Protein PseAAC
    pros_PseAAC = Proteins_PseAAC(pros)
    torch.save(pros_PseAAC, './dataset/training_dna/Proteins_PseAAC.pt')
    
    # Protein Secondary Structure
    pros_SS = Proteins_SS(pros_ss, conv2)
    torch.save(pros_SS, './dataset/training_dna/Proteins_SS.pt')
    
    # Aptamer 3Mer
    apts_3Mer = DNA_Aptamers_3Mer(apts, conv1)
    torch.save(apts_3Mer, './dataset/training_dna/Aptamers_3Mer.pt')
    
    # Aptamer PseKNC
    apts_PseKNC = DNA_Aptamers_PseKNC(apts)
    torch.save(apts_PseKNC, './dataset/training_dna/Aptamers_PseKNC.pt')
    
    # Aptamer Secondary Structure
    apts_SS = DNA_Aptamers_SS(apts, conv3)
    torch.save(apts_SS, './dataset/training_dna/Aptamers_SS.pt')
    
    # concatenate features
    Proteins_3Mer = torch.load('./dataset/training_dna/Proteins_3Mer.pt')
    Proteins_PseAAC = torch.load('./dataset/training_dna/Proteins_PseAAC.pt')
    Proteins_SS = torch.load('./dataset/training_dna/Proteins_SS.pt')
    
    Aptamers_3Mer = torch.load('./dataset/training_dna/Aptamers_3Mer.pt')
    Aptamers_PseKNC = torch.load('./dataset/training_dna/Aptamers_PseKNC.pt')
    Aptamers_SS = torch.load('./dataset/training_dna/Aptamers_SS.pt')
    
    features_tensor = torch.cat((Proteins_3Mer, Proteins_PseAAC, Proteins_SS, Aptamers_3Mer, Aptamers_PseKNC, Aptamers_SS), 1)
    torch.save(features_tensor, './dataset/training_dna/features_tensor.pt')
    
    targets_tensor = torch.tensor(data['Label'], dtype=torch.int32)
    torch.save(targets_tensor, './dataset/training_dna/targets_tensor.pt')
    
    # split train and test
    x_train, x_test, y_train, y_test = train_test_split(
    features_tensor, targets_tensor, test_size=0.1, random_state=42)
    
    torch.save(x_train, './dataset/training_dna/x_train.pt')
    torch.save(x_test, './dataset/training_dna/x_test.pt')
    torch.save(y_train, './dataset/training_dna/y_train.pt')
    torch.save(y_test, './dataset/training_dna/y_test.pt')
    
    