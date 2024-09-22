MMGCF：Motif-based generation of counterfactual explanations for molecules
===
Requirements
---

    1.pytorch 1.12.0
    2.torch-geometric 1.6.0
    3.rdkit 2020.09.1
    4.pandas 1.5.3
    5.numpy 1.24.3
To install RDKit, please follow the instructions here http://www.rdkit.org/docs/Install.html
Train RGCN
---

Train the RGCN by running:
        cd rgcn
        python data_preprocess.py
        python train_rgcn.py
Generate conterfactuals:
----
To generate counterfactual explanations for a specific sample, run:
        python train.py
        python fs_modify.py

