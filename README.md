# Neural-Architecture-Search-for-3D-Biomedical-Image-Classification
Neural Architecture Search for 3D Biomedical Image Classification

[Zeki Kuş](https://scholar.google.com/citations?user=h2B-3LwAAAAJ&hl=tr&oi=ao), [Berna Kiraz](https://scholar.google.com/citations?user=Je4hzioAAAAJ&hl=tr&oi=ao), [Musa Aydin](https://scholar.google.com/citations?user=yfKMO-wAAAAJ&hl=tr&oi=ao), [Alper Kiraz](https://scholar.google.com/citations?user=ic55Pj0AAAAJ&hl=tr)

This study introduces a novel extension of the PBC-NAS method for 3D medical image classification, aiming to balance prediction accuracy and model complexity. We focus on optimizing neural network architectures using Neural Architecture Search for six different 3D datasets from MedMNIST3D, including OrganMNIST3D, NoduleMNIST3D, FractureMNIST3D, AdrenalMNIST3D, VesselMNIST3D and SynapseMNIST3D. We have compared our method with state-of-the-art handcrafted networks, AutoML frameworks and recent NAS studies in terms of prediction performance and model complexity. The proposed NAS methods demonstrate superior performance compared to state-of-the-art handcrafted networks, AutoML frameworks, and other NAS studies. Our proposed model(Ours \#3$^\dagger$) achieves the highest average Area Under the Curve (AUC) of 0.915 and accuracy (ACC) of 0.847, outperforming all handcrafted networks, AutoML frameworks and recent NAS studies. Furthermore, all proposed models outperform handcrafted networks, AutoML frameworks and NAS studies in terms of average AUC and ACC (except for NAS). The study also highlights significant reductions in computational complexity, with FLOPs reduced by up to 45.51 times and parameters by up to 211 times compared to ResNet models. An ablation study reveals that while fine-tuning a model optimized for one dataset can achieve competitive results on other datasets, dataset-specific NAS is crucial for optimal performance. Despite this, the ablation results still outperform ResNets and AutoML frameworks in terms of average AUC and ACC. The study concludes that the proposed NAS approach effectively optimizes neural network architectures for complex 3D medical image classification tasks, achieving state-of-the-art performance without data augmentation.

## Reproducibility
* Code completely designed using Pytorch
* Explore train and test scripts for our study. Using the following code files, you can reproduce reported results with shared weight in [Results](results)
    * For Training:
        * [`train_pytorch.py`](train_pytorch.py)
    * For Testing:
        * [`test_pytorch.py`](test_pytorch.py)
