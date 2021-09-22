import os, datetime, argparse
import numpy as np
from Bio import pairwise2
from model import *

# Set these paths to your own paths
# Note that in UR90, 'uniref90' (last of the variable) is the prefix of the built database files in the database folder. Same does HHDB.
UR90 = "/bigdat1/pub/yuanqm/uniref90_2018_06/uniref90"
HHDB = "/bigdat1/pub/uniclust30/uniclust30_2017_10/uniclust30_2017_10"

Software_path = "/data2/users/yuanqm/PPI/Software/"
PSIBLAST = Software_path + "ncbi-blast-2.10.1+/bin/psiblast"
HHBLITS = Software_path + "hhsuite-3.0.3/bin/hhblits"
DSSP = Software_path + "dssp-3.1.4/mkdssp"
script_path = os.path.split(os.path.realpath(__file__))[0] + "/"
model_path = os.path.dirname(script_path[0:-1]) + "/Model/"

aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
aa_abbr = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
aa_dict = dict(zip(aa, aa_abbr))

Max_pssm = np.array([8, 9, 9, 9, 12, 10, 8, 8, 12, 9, 7, 9, 12, 10, 9, 8, 9, 13, 11, 8])
Min_pssm = np.array([-12, -12, -13, -13, -12, -11, -12, -12, -12, -12, -12, -12, -12, -12, -13, -12, -12, -13, -11, -12])
Max_hhm = np.array([12303, 12666, 12575, 12045, 12421, 12301, 12561, 12088, 12241, 11779, 12921, 12198, 12640, 12414, 12021, 11692, 11673, 12649, 12645, 12291])
Min_hhm = np.zeros(20)
Max_single = np.load(script_path + "Max_alphafold_single.npy")
Min_single = np.load(script_path + "Min_alphafold_single.npy")
threshold = {"both": 0.28, "single": 0.27, "evo": 0.28}


def get_seq(path, ID):
    seq = ""
    current_pos = -1000
    with open(path + ID + ".pdb", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0:4] == "ATOM" and int(line[22:26].strip()) != current_pos:
            aa_type = line[17:20].strip()
            seq += aa_dict[aa_type]
            current_pos = int(line[22:26].strip())
    return seq


def process_distance_map(distance_map_file):
    with open(distance_map_file, "r") as f:
        lines = f.readlines()

    seq = lines[0].strip()
    length = len(seq)
    distance_map = np.zeros((length, length))

    if lines[1][0] == "#": # missed residues
        missed_idx = [int(x) for x in lines[1].split(":")[1].strip().split()] # 0-based
        lines = lines[2:]
    else:
        missed_idx = []
        lines = lines[1:]

    for i in range(0, len(lines)):
        record = lines[i].strip().split()
        for j in range(0, len(record)):
            distance_map[i + 1][j] = float(record[j])

    for idx in missed_idx:
        if idx > 0:
            distance_map[idx][idx - 1] = 3.8
        if idx > 1:
            distance_map[idx][idx - 2] = 5.4
        if idx < length - 1:
            distance_map[idx + 1][idx] = 3.8
        if idx < length - 2:
            distance_map[idx + 2][idx] = 5.4

    distance_map = distance_map + distance_map.T
    return seq, distance_map


def get_distance_map(data_path, ID, PDB_seq):
    os.system("{}caldis_CA {}.pdb > {}.map".format(script_path, data_path + ID, data_path + ID))
    dis_map_seq, dis_map = process_distance_map(data_path + ID + ".map")
    if PDB_seq != dis_map_seq:
        raise Exception("PDB_seq & dismap_seq mismatch")
    else:
        np.save(data_path + ID + "_dismap.npy", dis_map)
        return 0


def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
        SS_vec[SS_type.find(SS)] = 1
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, dssp_feature


def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
    SS_vec[-1] = 1
    padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


def transform_dssp(dssp_feature):
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

    return dssp_feature


def get_dssp(data_path, ID, ref_seq):
    os.system("{} -i {}.pdb -o {}.dssp".format(DSSP, data_path + ID, data_path + ID))
    dssp_seq, dssp_matrix = process_dssp(data_path + ID + ".dssp")
    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)

    np.save(data_path + ID + "_dssp.npy", transform_dssp(dssp_matrix))
    return 0


def process_pssm(pssm_file):
    with open(pssm_file, "r") as f:
        lines = f.readlines()
    pssm_feature = []
    for line in lines:
        if line == "\n":
            continue
        record = line.strip().split()
        if record[0].isdigit():
            pssm_feature.append([int(x) for x in record[2:22]])
    pssm_feature = (np.array(pssm_feature) - Min_pssm) / (Max_pssm - Min_pssm)

    return pssm_feature


def process_hhm(hhm_file):
    with open(hhm_file, "r") as f:
        lines = f.readlines()
    hhm_feature = []
    p = 0
    while lines[p][0] != "#":
        p += 1
    p += 5
    for i in range(p, len(lines), 3):
        if lines[i] == "//\n":
            continue
        feature = []
        record = lines[i].strip().split()[2:-1]
        for x in record:
            if x == "*":
                feature.append(9999)
            else:
                feature.append(int(x))
        hhm_feature.append(feature)
    hhm_feature = (np.array(hhm_feature) - Min_hhm) / (Max_hhm - Min_hhm)

    return hhm_feature


def MSA(data_path, ID):
    os.system("{0} -db {1} -num_iterations 3 -num_alignments 1 -num_threads 4 -query {2}{3}.fa -out {2}{3}.bla -out_ascii_pssm {2}{3}.pssm".format(PSIBLAST, UR90, data_path, ID))
    os.system("{0} -i {1}{2}.fa -ohhm {1}{2}.hhm -oa3m {1}{2}.a3m -d {3} -v 0 -maxres 40000 -cpu 6 -Z 0 -o {1}{2}.hhr".format(HHBLITS, data_path, ID, HHDB))
    pssm_matrix = process_pssm(data_path + ID + ".pssm")
    np.save(data_path + ID + "_pssm", pssm_matrix)
    hhm_matrix = process_hhm(data_path + ID + ".hhm")
    np.save(data_path + ID + "_hhm", hhm_matrix)


def feature_extraction(path, ID, msa):
    PDB_seq = get_seq(path, ID)

    if msa == "both" or msa == "single":
        single = np.load(path + ID + "_single.npy")
        if len(single) != len(PDB_seq):
            return "The length of the protein dosen't match the length of the single representation matrix"
        single = (single - Min_single) / (Max_single - Min_single)
        np.save(path + ID + "_single_norm.npy", single)
    if msa == "both" or msa == "evo":
        with open(path + ID + ".fa", "w") as f:
            f.write(">" + ID + "\n" + PDB_seq)
        MSA(path, ID)
    try:
        get_dssp(path, ID, PDB_seq)
        get_distance_map(path, ID, PDB_seq)
        return 0
    except Exception as ex:
        return ex


def main(path, ID, msa):
    print("\nFeature extraction begins at {}.\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))
    error_code = feature_extraction(path, ID, msa)
    if error_code != 0:
        print("Error info: {}. Please make sure you provide the correct input or contact the authors of GraphSite.".format(error_code))
    else:
        print("\nFeature extraction is done at {}.\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))
        print("Predicting...\n")
        outputs = predict_one_protein(path, model_path + msa + "/", ID, msa)
        pred_scores = [round(score, 4) for score in outputs]

        GraphSite_threshold = threshold[msa]
        binary_preds = [1 if score >= GraphSite_threshold else 0 for score in pred_scores]

        seq = get_seq(path, ID)

        # Final prediction results
        res = "The threshold of the predictive score to determine protein-DNA binding sites is set to {}.\n".format(GraphSite_threshold)
        res += "AA\tProb\tPred\n"
        for i in range(len(seq)):
            res += (seq[i] + "\t" + str(pred_scores[i]) + "\t" + str(binary_preds[i]) + "\n")

        with open(path + ID + "_" + msa + "_results.txt", "w") as f:
            f.write(res)
        print("Results are saved in {}!".format(path + ID + "_" + msa + "_results.txt"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, help = "Query data path")
    parser.add_argument("--id", type = str, help = "ID of the protein")
    parser.add_argument("--msa", type = str, default = "both", help = "Source of the MSA information (single, evo or both)")
    args = parser.parse_args()
    args.path += "/"

    if args.msa not in ["both", "single", "evo"]:
        print("Invalid --msa value!")
    elif not os.path.exists(args.path + args.id + ".pdb"):
        print("Please provide the AlphaFold2-predicted structure under {} !".format(args.path))
    elif args.msa != "evo" and not os.path.exists(args.path + args.id + "_single.npy"):
        print("Please provide the single representation from AlphaFold2 under {} !".format(args.path))
    else:
        main(args.path, args.id, args.msa)
