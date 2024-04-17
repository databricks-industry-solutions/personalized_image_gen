<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-CHANGE_ME-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/CHANGE_ME.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-CHANGE_ME-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
[![POC](https://img.shields.io/badge/POC-CHANGE_ME-green?style=for-the-badge)](https://databricks.com/try-databricks)

## Business Problem
Recent advancements in large text-to-image models have shown unparalleled capabilities, enabling high-quality generation of images in diverse contexts based on natural language prompts or existing images. Today, design professionals across various industries are harnessing these models to generate images that serve as inspiration for their next product designs. With or without further refinement, the generated images may even be used as initial prototypes.

Like any other generative models, tailoring the output content is crucial for building a successful application. For example, fashion designers may want to generate images of an item reflecting the style of a specific collection. Furniture designers may want to see famous designer chairs made from different materials. Customization is necessary for such use cases. However, pre-trained text-to-image models available today often lack the capacity to accurately generate specific subjects in various contexts. Thus, fine-tuning the models on the  specific subject images becomes essential.

This solution accelerator provides users with a tool to expedite the end-to-end development of personalized image generation models. The asset including a series of notebooks demonstrates how to preprocess subject images, how to fine-tune a text-to-image diffusion model, how to manage the fine-tuned model, and how to make that model available for downstream applications by deploying it behind a real-time endpoint. The solution is by design customizable (bring your own images) and scalable leveraging the powerful distributed compute infrastructure of Databricks.

The solution uses DreamBooth to fine-tune Stable Diffusion XL using sample images featuring designer chairs.


## Why Stable Diffusion XL?
[Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/) is a powerful open-source text-to-image model available for commercial usage. Developed by Stability AI, its weights are publicly accessible via Hugging Face, which has native support on Databricks. Both the off-the-shelf and the fine-tuned versions of the model are widely adopted by companies and are utilized for their mission-critical applications.


## Why DreamBooth?
[Dreambooth](https://dreambooth.github.io/) is an innovative technique developed by researchers from Google Research and Boston University that enables fine-tuning of text-to-image models using only a few images depicting a particular subject or style. Through fine-tuning the models acquire the ability to generate images of the subject/s in diverse contexts. DreamBooth is seamlessly integrated into Hugging Face’s Diffusers library, with its training scripts readily available to the public.


## Why Databricks Mosaic AI?
[Databricks Mosaic AI](https://www.databricks.com/product/machine-learning) offers a great option for GenAI project development and management. It provides a scalable unified platform scoping Data and AI. Data needed to train models is readily available via [Unity Catalog Volumes](https://www.databricks.com/product/unity-catalog) and GenAI models can be easily managed and deployed using [MLflow](https://www.databricks.com/product/managed-mlflow).


## Getting Started
This project is structured in 4 notebooks.  

The first notebook, [01_introduction](https://github.com/databricks-industry-solutions/personalized_image_gen/blob/main/01_introduction.py), walks you through how to download Stable Diffusion XL from Hugging Face and generate an image conditioned on a simple prompt. This notebook is aimed to demonstrate how easy it is to use an open source image generation model off-the-shelf on Databricks. 

The second notebook,  [02_data_prep](https://github.com/databricks-industry-solutions/personalized_image_gen/blob/main/02_data_prep.py), downloads a sample training dataset consisting of 25 images of designer chairs from the same repository and applies preprocessing. The main step performed here is to annotate each image with a unique token referring to the subject and a context. We use Unity Catalog Volumes to manage the preprocessed and post-processed images. 

The third notebook, [03_fine_tuning](https://github.com/databricks-industry-solutions/personalized_image_gen/blob/main/03_fine_tuning.py), shows how to fine-tune Stable Diffusion XL using DreamBooth. Here, we combine techniques such as mixed precision and LoRA to make the training efficient and to reduce the memory footprint. The second part of the notebook takes the fine-tuned model and registers it to Unity Catalog using MLflow. 
 
The final notebook, [04_deploy_model](https://github.com/databricks-industry-solutions/personalized_image_gen/blob/main/04_deploy_model.py), takes the model registered in Unity Catalog and deploys it behind Databricks Mosaic AI Model Serving endpoint. This allows end users to send an image generation request and get the results back in real time via Rest API.  

To get started, simply clone this repository to your Databricks Repos and run the notebooks in the right sequence. For the compute, we recommend a single node cluster with multiple A10 or A100 GPU instances (see the second notebook for more details). In order to use your own images for fine tuning, follow the instructions in the notebook, 02_data_prep. 

Note that DreamBooth is sensitive to hyperparameters, and it is known to easily overfit. For detailed explanation of the limitations and how to deal with them, read the original [paper](https://arxiv.org/abs/2208.12242) or this [blog post](https://huggingface.co/blog/dreambooth). 

## Reference Architecture
<img src='https://github.com/databricks-industry-solutions/personalized_image_gen/blob/main/images/architecture/architecture.png' width=800>

## Authors
<ryuta.yoshimatsu@databricks.com>

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 

## License

&copy; 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
