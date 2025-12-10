# architecture of this project

* evaluation.py: 实现sliced wasserstein distance
* model.py 用diffusers的1dUNet构建ddpm，实现cosine与sigmoid scheduler
* train_ecg.py 训练标准ddpm，metric为sliced Wasserstein distance
* model_bilevel.py 用diffusers的1dUnet构建ddpm，并且将scheduler参数化
* train_ecg_bilevel.py 训练bilevel diffusion