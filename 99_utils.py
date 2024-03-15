# Databricks notebook source
# Installing requirement libraries
%pip install -r ./requirements.txt --quiet
dbutils.library.restartPython()

# COMMAND ----------

# Common imports used throughout.
import matplotlib.pyplot as plt
import PIL
import torch

# COMMAND ----------


def show_image(image: PIL.Image.Image):
    """
    Show one generated image.
    """
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def show_image_grid(imgs, rows, cols, resize=256):
    """
    Show multiple generated images in grid.
    """
    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = PIL.Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def caption_images(input_image, blip_processor, blip_model, device):
    """
    Caption images with an annotation model.
    """
    inputs = blip_processor(images=input_image, return_tensors="pt").to(
        device, torch.float16
    )
    pixel_values = inputs.pixel_values
    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return generated_caption
