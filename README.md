# Intro  
GraphSite is a novel framework for sequence-based protein-DNA binding site prediction using graph transformer and predicted protein structures from AlphaFold. This repository is still under development!  
![GraphSite_framework](https://github.com/biomed-AI/GraphSite/blob/master/IMG/GraphSite_framework.png)

# System requirement  
GraphSite is developed under Linux environment with:  
python  3.8.5  
numpy  1.19.1  
pandas  1.1.3  
torch  1.7.1  
scikit-learn  0.23.2  

# Run GraphSite for training and testing  
Train:  
```
python main.py --train --seed 2021 --run_id demo
```
Test on Test_129:  
```
python main.py --test1 --seed 2021 --run_id demo
```
Test on Test_196:  
```
python main.py --test2 --seed 2021 --run_id demo
```

# Dataset, feature and model  
We provide the datasets, the pre-predicted structures, part of the pre-computed features (just for demonstration) and the pre-trained models here for those interested in reproducing our paper.  
The datasets used in this study (DNA_Train_573, DNA_Test_129 and DNA_Test_196) are stored in ./Dataset in python dictionary format:  
```
Dataset[ID] = [seq, label]
```
The distance maps (max_len * max_len), the normalized node feature matrixes (max_len * 398), label (max_len,) and the padding mask (max_len,) are stored in ./Feature/input/ where max_len = 1937.
The pre-trained GraphSite models can be found under ./output/demo/.  

# Contact   
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)
