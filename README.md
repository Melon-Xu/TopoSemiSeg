# [TopoSemiSeg: Enforcing Topological Consistency for Semi-Supervised Segmentation of Histopathology Images](https://arxiv.org/abs/2311.16447)
The official implementation of "TopoSemiSeg: Enforcing Topological Consistency for Semi-Supervised Segmentation of Histopathology Images". Code will be coming soon.

## Environment
Training and evaluation environment: Python 3.9.12, PyTorch 2.0.1, CUDA 11.7.
We use CUbicalRipser (cripser) to extract topological features.
Run the following command to install required packages.
```
pip install -r requirements.txt
```

## Inituition of Decomposition & Matching
<p align="center">
  <img src="./figures/inituition.jpg" alt="drawing", width="850"/>
</p>

## Overall Framework
<p align="center">
  <img src="./figures/overall_framework.jpg" alt="drawing", width="850"/>
</p>

## Qualitative Results
<p align="center">
  <img src="./figures/qualitative_results.jpg" alt="drawing", width="850"/>
</p>

## Citation
```bibtex
@InProceedings{xu2023toposemiseg,
    author    = {Xu, Meilong and Hu, Xiaoling and Gupta, Saumya and Abousamra, Shahira and Chen, Chao},
    title     = {TopoSemiSeg: Enforcing Topological Consistency for Semi-Supervised Segmentation of Histopathology Images},
    journal   = {arXiv},
    year      = {2023},
    volume    = {abs/2311.16447},
}
