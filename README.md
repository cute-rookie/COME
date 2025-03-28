# Conditional Diffusion Model with Feature Decoupling for Fair Semi-supervised Retrieval

This repository contains a PyTorch implementation for our paper "Conditional Diffusion Model with Feature Decoupling for Fair Semi-supervised Retrieval"

## Getting Started

### Main Usage

Use the following command to run the main script with configuration options:

```
python main_utkface_age_ethnicity.py
python main_utkface_ethnicity_age.py
python main_celeba.py
```

We support two datasets [UTK Face](https://susanqq.github.io/UTKFace/) and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), place the downloaded dataset in the './data' of the root folder.

The structure of the datasets is:

```
For UTKFace
-data
--UTKFace
---100_0_0_20170112213500903.jpg
---100_0_0_20170112215240346.jpg
......
For CelebA
-data
--celeba
---Anno
----identity_CelebA.txt
----list_attr_celeba.txt
----list_bbox_celeba.txt
----list_landmarks_align_celeba.txt
----list_landmarks_celeba.txt
---Eval
----list_eval_partition.txt
---Img
----img_align_celeba
-----000001.jpg
-----000002.jpg
......
```

Our code is based on the deformable attention, so you need to put the [ops](https://drive.google.com/drive/folders/1kj6Uw5Ox5AQwfx6xU0fJ9ZNduZEioQIT?usp=sharing) folder into the '. /models' folder.

# About LLM

We used some LLMs for image generation and pseudo-labeling. Considering that LLMs are very resource-consuming, we put all the generated or labeled images [here](https://drive.google.com/drive/folders/1kj6Uw5Ox5AQwfx6xU0fJ9ZNduZEioQIT?usp=sharing). You can download and unzip them directly into '. /dataset'. In addition, '. /processed_files' folder stores the similarity matrix calculated by the CLIP model, which can also be used directly after downloading.

## Diffusion Model

We used a diffusion model for sample generation, and we will upload the trained LoRA structure later. But this does not affect the code inference. You can download the [diffusion model](https://modelscope.cn/collections/Stable-Diffusion-35-8cd8a1c210b84a) and put it in the '. /LLM' folder.

## CLIP

We used the CLIP model to construct the similarity between images and text, you can download the pre-trained model [here](https://huggingface.co/jinaai/jina-clip-v2/tree/main).

## Qwen-VL

We used the QWen-VL model for pseudo-labeling, you can download the pre-trained model [here](https://huggingface.co/Qwen). Due to resource constraints, we only downloaded the QWen2.5 VL 7B model and used api to call the 72B model.