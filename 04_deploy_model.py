# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/personalized_image_gen).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Create a model serving endpoint with Python
# MAGIC Now we have a fine-tuned model registered in Unity Catalog, our final step is to deploy this model behind a Model Serving endpoint. This notebook covers wrapping the REST API queries for model serving endpoint creation, updating endpoint configuration based on model version, and endpoint deletion with Python for your Python model serving workflows.

# COMMAND ----------

import mlflow

# Set the registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md Specify some variables.

# COMMAND ----------

theme = "chair"
catalog = "sdxl_image_gen"
log_schema = "log" # A schema within the catalog where the inferece log is going to be stored 
model_name = f"{catalog}.model.sdxl-fine-tuned-{theme}"  # An existing model in model registry, may have multiple versions
model_serving_endpoint_name = f"sdxl-fine-tuned-{theme}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up configurations
# MAGIC Depending on the latency and throughput requirements of your use case, you want to choose the right `workload_type` and `workload_size`. **Note that if you're using Azure Databricks, use `GPU_LARGE` for `workload_type`**. The `auto_capture_config` block specifies where to write the inference logs: i.e. requests and responses from the endpoint with a timestamp. 

# COMMAND ----------

# Get the champion model version
champion_version = client.get_model_version_by_alias(model_name, "champion")
model_version = champion_version.version

my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                "workload_type": "GPU_MEDIUM",
                "workload_size": "Small",
                "scale_to_zero_enabled": "false",
            }
        ],
        "auto_capture_config": {
            "catalog_name": catalog,
            "schema_name": log_schema,
            "table_name_prefix": model_serving_endpoint_name,
        },
    },
}

# Make sure to the schema for the inference table exists
_ = spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {catalog}.{log_schema}"
)

# Make sure to drop the inference table of it exists
_ = spark.sql(
    f"DROP TABLE IF EXISTS {catalog}.{log_schema}.`{model_serving_endpoint_name}_payload`"
)

# COMMAND ----------

# MAGIC %md
# MAGIC The following defines Python functions that:
# MAGIC - create a model serving endpoint
# MAGIC - update a model serving endpoint configuration with the latest model version
# MAGIC - delete a model serving endpoint

# COMMAND ----------

import mlflow.deployments

def func_create_endpoint(json):
    client = mlflow.deployments.get_deploy_client("databricks")
    try:
        # Check if the endpoint already exists
        client.get_deployment(json["name"])
        # Update the existing endpoint with the new model version
        client.update_deployment(
            name=json["name"], 
            config=json["config"]
        )
    except:
        # Create a new endpoint if it doesn't exist
        client.create_endpoint(
            name = model_serving_endpoint_name,
            config = json["config"],
        )

def func_delete_model_serving_endpoint(json):
    client = mlflow.deployments.get_deploy_client("databricks")
    # Delete the specified endpoint
    client.delete_endpoint(json["name"])
    print(json["name"], "endpoint is deleted!")

# COMMAND ----------

func_create_endpoint(my_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for the endpoint to be ready
# MAGIC
# MAGIC The `wait_for_endpoint()` function defined in the following command gets and returns the serving endpoint status.

# COMMAND ----------

def wait_for_endpoint(endpoint_name):
    '''Wait for a model serving endpoint to be ready'''
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
    import time

    # Initialize WorkspaceClient
    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(endpoint_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {endpoint_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
            print('endpoint ready.')
            return
        else:
            break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

wait_for_endpoint(my_json["name"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the model
# MAGIC The following command defines the `score_model()` function  and an example scoring request under the `payload_json` variable.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

def generate_image(endpoint, dataset):
    # Initialize the MLflow deployment client for Databricks
    client = get_deploy_client("databricks")
    
    # Convert the dataset to a dictionary in 'split' orientation
    ds_dict = {"dataframe_split": dataset.to_dict(orient="split")}
    
    # Make a prediction request to the specified endpoint with the dataset
    response = client.predict(endpoint=endpoint, inputs=ds_dict)
    
    return response

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the prompt and number of inference steps
prompt = pd.DataFrame(
    {"prompt": ["A photo of an orange bcnchr chair"], "num_inference_steps": 25}
)

# Generate image using the specified endpoint and prompt
t = generate_image(my_json["name"], prompt)

# Display the generated image
plt.imshow(t["predictions"])
plt.axis("off")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete the endpoint

# COMMAND ----------

func_delete_model_serving_endpoint(my_json)

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | bitsandbytes | Accessible large language models via k-bit quantization for PyTorch. | MIT | https://pypi.org/project/bitsandbytes/
# MAGIC | diffusers | A library for pretrained diffusion models for generating images, audio, etc. | Apache 2.0 | https://pypi.org/project/diffusers/
# MAGIC | stable-diffusion-xl-base-1.0 | A model that can be used to generate and modify images based on text prompts. | CreativeML Open RAIL++-M License | https://github.com/Stability-AI/generative-models
