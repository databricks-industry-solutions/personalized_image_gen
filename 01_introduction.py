# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/personalized_image_gen).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Creating Brand-Aligned Images Using Generative AI
# MAGIC Design professionals across various industries are harnessing diffusion models to generate images that serve as inspiration for their next product designs. This solution accelerator provides Databricks users with a tool to expedite the end-to-end development of personalized image generation applications. The asset including a series of notebooks demonstrates how to preprocess training images, fine-tune a text-to-image diffusion model, manage the fine-tuned model, and deploy the model behind an endpoint to make it available for downstream applications. The solution is by design customizable (bring your own images) and scalable leveraging Databricks powerful distributed compute.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster configuration
# MAGIC We recommend using a cluster with the following specifications to run this solution accelerator:
# MAGIC - Unity Catalog enabled cluster 
# MAGIC - Databricks Runtime 14.3LTS ML or above
# MAGIC - Single-node multi-GPU cluster: e.g. `g5.48xlarge` on AWS or `Standard_NC48ads_A100_v4` on Azure Databricks.

# COMMAND ----------

# DBTITLE 1,Install requirements and load helper functions
# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use off-the-shelf Stable Diffusion XL Base model

# COMMAND ----------

# MAGIC %md
# MAGIC Stable Diffusion XL Base is one of the most powerful open-source text-to-image models available for commercial usage (as of March 1, 2024). The model was developed by Stability AI, and its weights are publicly accessible via Hugging Face, which has native support on Databricks. To download this model from Hugging Face and generate an image, run the following cells. 

# COMMAND ----------

import torch
from diffusers import DiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.to(device)

# COMMAND ----------

prompt = "A photo of a brown leather chair in a living room."
image = pipe(prompt=prompt).images[0]
show_image(image) # This function is defined in 99_utils notebook

# COMMAND ----------

# MAGIC %md We free up some memory to be able to run the following notebooks.

# COMMAND ----------

import gc

# delete the pipeline and free up some memory
del pipe
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | bitsandbytes | Accessible large language models via k-bit quantization for PyTorch. | MIT | https://pypi.org/project/bitsandbytes/
# MAGIC | diffusers | A library for pretrained diffusion models for generating images, audio, etc. | Apache 2.0 | https://pypi.org/project/diffusers/
# MAGIC | stable-diffusion-xl-base-1.0 | A model that can be used to generate and modify images based on text prompts. | CreativeML Open RAIL++-M License | https://github.com/Stability-AI/generative-models
