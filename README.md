# Intro  
GraphSite is a novel framework for sequence-based protein-DNA binding site prediction using graph transformer and predicted protein structures from AlphaFold2. We recommend you to use the [web server (new version)](http://bio-web1.nscc-gz.cn/app/graphsite) of GraphSite if your input is small.  
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
Besides, you need to provide the predicted protein structures along with the single representations from AlphaFold2. To generate these files from sequences, you can first run [AlphaFold2](http://bio-web1.nscc-gz.cn/app/alphaFold2_bio) on our biomedical AI platform. You can also visit [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) to directly download the predicted structures and single representations (coming soon).  
However, if you use the reduced version of GraphSite, the BLAST+&HH-suite and AlphaFold2 single representations are alternative.  
:star:Note: If you run the standalone AlaphaFold2 on your own, please set `return_representations=True` in `class AlphaFold(hk.Module)`. Besides the predicted structures, the outputs of AlphaFold2 also contain `result_model_*.pkl`, which is a python dictionary. If you set this parameter, you can get the single representation matrix in this dictionary via the keys of "representations" and then "single".  

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

# Dataset and model  
We provide the datasets, the pre-predicted structures, the single representations, and the pre-trained models here for those interested in reproducing our paper.  
The datasets used in this study (DNA_Train_573, DNA_Test_129 and DNA_Test_181) are stored in ./Dataset/ in fasta format.  
The AlphaFold2-predicted structures of the proteins in these three datasets are also in ./Dataset/.  
The AlphaFold2 single representations of the proteins can be found in [here](https://drive.google.com/file/d/1qCbqAncR1k6IXmTIPhaLEyWxuk1Wy-z-/view?usp=sharing).  
The pre-trained GraphSite models can be found under ./Model/.  

# Citation and contact  
Citation:  
```bibtex
@article{10.1093/bib/bbab564,
    author = {Yuan, Qianmu and Chen, Sheng and Rao, Jiahua and Zheng, Shuangjia and Zhao, Huiying and Yang, Yuedong},
    title = "{AlphaFold2-aware protein–DNA binding site prediction using graph transformer}",
    journal = {Briefings in Bioinformatics},
    volume = {23},
    number = {2},
    year = {2022},
    month = {01},
    issn = {1477-4054},
    doi = {10.1093/bib/bbab564},
    url = {https://doi.org/10.1093/bib/bbab564},
}
```

Contact:  
Qianmu Yuan (yuanqm3@mail3.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)
