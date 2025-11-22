# <img src="doc/icon.png" style="width:40px;height:auto">SUICA

[**SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics**](https://arxiv.org/)<br />
[Qingtian Zhu](https://qtzhu.me)<sup>1*</sup>, [Yumin Zheng](https://scholar.google.com/citations?user=NQTC8NMAAAAJ&hl=en)<sup>2,3*</sup>, [Yuling Sang](https://scholar.google.com/citations?user=qfkckBkAAAAJ&hl=zh-CN)<sup>4</sup>, [Yifan Zhan](https://yifever20002.github.io)<sup>1</sup>, Ziyan Zhu<sup>5</sup>, [Jun Ding](https://meakinsmcgill.com/ding/)<sup>2,3,6</sup>, [Yinqiang Zheng](https://www.ai.u-tokyo.ac.jp/ja/members/yqzheng)<sup>1</sup><br />
<sup>1</sup>The University of Tokyo, <sup>2</sup>McGill University, <sup>3</sup>MUHC Research Institute, <sup>4</sup>Duke-NUS Medical School,<br /> 
<sup>5</sup>Carnegie Mellon University, <sup>6</sup>Mila-Quebec AI Institute<br />
**ICML 2025**
<p align="center">
    <img src="doc/SUICA_method_fig.png" width="800"><br>SUICA pipeline
</p>

## Environment

```shell
conda create -n SUICA python=3.9 -y && conda activate SUICA
pip install -r requirements.txt
```
## To Run Your Data

The typical data structure is as follows:
```
|-- custom_root_path
    |-- configs #configuration file for GAE training and GAE-INR joint training
        |-- ST
            |--embedder_gae.yaml
            |--inr_embd.yaml
    |-- data 
        |--preprocessed_data
            |--your_data.h5ad
    |logs
    |--networks
    |--scripts
    |--systems
    |--datasets.py  
    |--train.py # The main file
    |--utils.py
```

## Training

**Train the Graph AutoEncoder (GAE)**
```
python train.py --mode embedder --conf ./configs/ST/embedder_gae.yaml
```

**Train the GAE-INR**
```
python train.py --mode inr --conf ./configs/ST/inr_embd.yaml
```

## Citation
```
@InProceedings{pmlr-v267-zhu25af,
  title = 	 {{SUICA}: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics},
  author =       {Zhu, Qingtian and Zheng, Yumin and Sang, Yuling and Zhan, Yifan and Zhu, Ziyan and Ding, Jun and Zheng, Yinqiang},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages = 	 {80448--80462},
  year = 	 {2025},
  editor = 	 {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume = 	 {267},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhu25af/zhu25af.pdf},
  url = 	 {https://proceedings.mlr.press/v267/zhu25af.html},
  abstract = 	 {Spatial Transcriptomics (ST) is a method that captures gene expression profiles aligned with spatial coordinates. The discrete spatial distribution and the super-high dimensional sequencing results make ST data challenging to be modeled effectively. In this paper, we manage to model ST in a continuous and compact manner by the proposed tool, SUICA, empowered by the great approximation capability of Implicit Neural Representations (INRs) that can enhance both the spatial density and the gene expression. Concretely within the proposed SUICA, we incorporate a graph-augmented Autoencoder to effectively model the context information of the unstructured spots and provide informative embeddings that are structure-aware for spatial mapping. We also tackle the extremely skewed distribution in a regression-by-classification fashion and enforce classification-based loss functions for the optimization of SUICA. By extensive experiments of a wide range of common ST platforms under varying degradations, SUICA outperforms both conventional INR variants and SOTA methods regarding numerical fidelity, statistical correlation, and bio-conservation. The prediction by SUICA also showcases amplified gene signatures that enriches the bio-conservation of the raw data and benefits subsequent analysis.}
}

```
