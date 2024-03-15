# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions).

# COMMAND ----------

# DBTITLE 1,Install requirements and load helper functions
# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC #Prepare your images for fine-tuning
# MAGIC  Tailoring the output of a generative model is crucial for building a successful application. This applies to use cases powered by image generation models as well. Imagine a scenaio where a furniture designer seeks to generate images for ideation purposes and they want their old products to be reflected on the generated images. Not only that but they also want to see some variations, for example, in material or color. In such instances, it is imperative that the model knows their old products and can apply some new styles on them. Customization is necessary in a case like this. We can do this by fine-tuning a pre-trained model on our own images.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage your images in Unity Catalog Volumes

# COMMAND ----------

# MAGIC %md #LINK TO GITHUB LOCATION NEEDED

# COMMAND ----------

# MAGIC %md
# MAGIC This solution accelerator uses 25 training images stored in the subfolders of ```/images/chair/``` to fine-tune a model. If you have imported this accelerator from GitHub, the images should already be in place.  If you simply downloaded the notebooks, you will need to create the folder structure in your workspace and import the images from **THIS LOCATION** for the following cells to work without modification.

# COMMAND ----------

theme = "chair"
catalog = "sdxl_image_gen" # Name of the catalog we use to manage our assets (e.g. images, weights, datasets) 
volumes_dir = f"/Volumes/{catalog}/{theme}" # Path to the directories in UC Volumes

# COMMAND ----------

# Make sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}") 
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{theme}") 

# COMMAND ----------

import os
import subprocess

# Create volumes under the schma, and copy the training images into it 
for volume in os.listdir("./images/chair"):
  volume_name = f"{catalog}.{theme}.{volume}"
  spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_name}")
  command = f"cp ./images/chair/{volume}/*.jpg /Volumes/{catalog}/{theme}/{volume}/"
  process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
  output, error = process.communicate()
  if error:
    print('Output: ', output)
    print('Error: ', error)

# COMMAND ----------

import glob

# Display images in Volumes
img_paths = f"{volumes_dir}/*/*.jpg"
imgs = [PIL.Image.open(path) for path in glob.glob(img_paths)]
num_imgs_to_preview = 25
show_image_grid(imgs[:num_imgs_to_preview], 5, 5) # Custom function defined in util notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Annotate your images with a unique token
# MAGIC For fine-tuning we need to provide a caption for each training image. 25 images above consists of 5 different styles of chairs. We assign a unique token for each style and use it in the caption: e.g. “A photo of a BCNCHR chair”, where BCNCHR is a unique token assigned to the black leather chair on the top row (see the output of the previous cell). The uniqueness of the token helps us preserve the syntactic and semantic knowledge that the base pre-trained model brings. The idea of fine-tuning is not to mess with what the model already knows, and learn the association between the new token and the subject. Read more about this [here](https://dreambooth.github.io/).
# MAGIC
# MAGIC In the following cells, we add a token (e.g. BCNCHR) to each caption using a caption prefix. For this example, we use the format: "a photo of a BCNCHR chair," but it could also be something like: "a photo of a chair in the style of BCNCHR".

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automate the generation of custom captions with BLIP
# MAGIC When we have too many training images, automating the caption generation using a model like BLIP is an option. 

# COMMAND ----------

import pandas as pd
import PIL
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

# load the processor and the captioning model
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16
).to(device)

# COMMAND ----------

# create a list of (Pil.Image, path) pairs
imgs_and_paths = [
    (path, PIL.Image.open(path).rotate(-90))
    for path in glob.glob(f"{volumes_dir}/*/*.jpg")
]

# COMMAND ----------

import json

captions = []
for img in imgs_and_paths:
    instance_class = img[0].split("/")[4].replace("_", " ")
    caption_prefix = f"a photo of a {instance_class} {theme}: "
    caption = (
        caption_prefix
        + caption_images(img[1], blip_processor, blip_model, device).split("\n")[0] # Function caption_images is defined in utils notebook 
    )
    captions.append(caption)

# COMMAND ----------

# Show the captions generated by BLIP
display(pd.DataFrame(captions).rename(columns={0: "caption"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage Dataset in UC Volumes
# MAGIC We create a Hugging Face Dataset object and store it in Unity Catalog Volume.

# COMMAND ----------

from datasets import Dataset, Image

d = {
    "image": [imgs[0] for imgs in imgs_and_paths],
    "caption": [caption for caption in captions],
}
dataset = Dataset.from_dict(d).cast_column("image", Image())
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{theme}.dataset")
dataset.save_to_disk(f"/Volumes/{catalog}/{theme}/dataset")

# COMMAND ----------

# MAGIC %md Let's free up some memory again.

# COMMAND ----------

import gc
del blip_processor, blip_model
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
