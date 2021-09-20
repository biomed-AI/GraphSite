# Intro  
GraphSite is a novel framework for sequence-based protein-DNA binding site prediction using graph transformer and predicted protein structures from AlphaFold2. We recommend you to use the [web server](https://biomed.nscc-gz.cn/apps/GraphSite) of GraphSite if your input is small (coming soon).  
![GraphSite_framework](https://github.com/biomed-AI/GraphSite/blob/master/IMG/GraphSite_framework.png)   

# System requirement  
GraphSite is developed under Linux environment with:  
python  3.8.5  
numpy  1.19.1  
pandas  1.1.3  
torch  1.7.1  
biopython  1.78  

# Software and database requirement  
To run the full & accurate version of GraphSite, you need to install the following three software and download the corresponding databases:  
[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
[DSSP](https://github.com/cmbi/dssp)  
Besides, you need to provide the predicted protein structures along with the single representations from AlphaFold2. We recommend you to register and use the [Tianhe-2 supercomputer](https://starlight.nscc-gz.cn) platform, where AlphaFold2 was already installed with graphical interface.  
However, if you use the reduced version of GraphSite, the BLAST+&HH-suite and AlphaFold2 single representations are alternative.  

# Build database and set path  
1. Use `makeblastdb` in BLAST+ to build UniRef90 ([guide](https://www.ncbi.nlm.nih.gov/books/NBK569841/)).  
2. Build Uniclust30 following [this guide](https://github.com/soedinglab/uniclust-pipeline).  
3. Set path variables `UR90`, `HHDB`, `PSIBLAST`, `HHBLITS` and `DSSP` in `GraphSite_predict.py`.  

# Run GraphSite for prediction  
Run full & accurate version of GraphSite:  
```
python ./script/GraphSite_predict.py --path ./demo/ --id 6ymw_B
```
This requires that the predicted structure `6ymw_B.pdb` and raw single representation `6ymw_B_single.npy` exist in the provided path.  
The program uses the full model in default. If you want to use the reduced version of GraphSite that adopts only AlphaFold2 single representation as MSA information, type as follows:  
```
python ./script/GraphSite_predict.py --path ./demo/ --id 6ymw_B --msa single
```
Set `--msa evo` to use only evolutionary features (PSSM + HMM) as MSA information (might causes large performance drop); Set `--msa both` to use the full version of GraphSite, which is the default option.  

# Dataset, feature and model  
We provide the datasets, the pre-predicted structures, the pre-computed features and the pre-trained models here for those interested in reproducing our paper.  
The datasets used in this study (DNA_Train_573, DNA_Test_129 and DNA_Test_181) are stored in ./Dataset/ in python dictionary format:  
```
Dataset[ID] = [seq, label]
```
The Min-Max normalized PSSM, HMM and DSSP feature matrixes are stored in ./Feature/ and the normalized AlphaFold2 single representations are stored in [google drive](https://drive.google.com/drive/folders/1GGqqYBZAd2IA5BuQEzsHVJon1ZiQbgEy?usp=sharing).  
The pre-trained GraphSite models can be found under ./Model/.  

# Contact  
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)
