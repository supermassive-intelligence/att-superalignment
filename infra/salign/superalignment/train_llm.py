from infra.salign.superalignment.text2sql import make_prompt as make_text2sql_prompt

from infra.salign.util.get_train_api_url import get_train_api_url

import scalarlm

import os
import time
import copy
import random

import logging

logger = logging.getLogger(__name__)


def train_llm(original_dataset):
    api_url = get_train_api_url()

    llm = scalarlm.SupermassiveIntelligence(api_url=api_url)

    dataset = copy.deepcopy(original_dataset)

    add_input_and_ouptput_fields(dataset)

    # Make sure that the dataset is globally shuffled
    random.seed(42)
    random.shuffle(dataset)

    dataset_size = len(dataset)
    max_steps = int(dataset_size * 1.5)

    status = llm.train(
        dataset,
        train_args={
            "max_steps": max_steps,
            "learning_rate": 3e-4,
            "gpus": 2,
            "timeout": 60 * 60 * 4 * 3,
            "max_token_block_size": 4096,
            "steps_per_checkpoint": 10000,
        },
    )

    job_hash = os.path.basename(status["job_status"]["job_directory"])
    tuned_model_name = status["job_status"]["model_name"]

    logger.info(
        f"Launched training job {job_hash} has status {status['job_status']['status']}"
    )

    status = wait_for_training_to_complete(llm, job_hash)

    return {
        "job_hash": job_hash,
        "model_name": tuned_model_name,
        "status": status["job_status"]["status"],
        "api_url": api_url,
    }


def wait_for_training_to_complete(llm, job_hash):
    while True:
        training_response = llm.get_training_job(job_hash)
        training_status = training_response["job_status"]["status"]
        logger.debug(f"Training job {job_hash} has status {training_status}")

        if training_status == "FAILED":
            raise RuntimeError(
                f"Training job {job_hash} has failed, please check the logs"
            )

        if training_status == "COMPLETED":
            break

        time.sleep(10)

    logger.info(
        f"Training job {job_hash} has completed successfully, waiting for model to be registered"
    )

    # 4. Wait for deployment of the pre-trained model
    training_response = llm.get_training_job(job_hash)

    while training_response["deployed"] is False:
        logger.debug(
            f"Model {job_hash} has not been registered yet, sleeping for 10 seconds"
        )
        time.sleep(10)
        training_response = llm.get_training_job(job_hash)

    logger.info(f"Model {job_hash} has been registered successfully")

    return training_response


def add_input_and_ouptput_fields(dataset):
    for example in dataset:
        example["input"] = make_input(example)
        example["output"] = make_output(example)


def make_input(example):
    return make_text2sql_prompt(example)


def make_output(example):
    return (
        example["reasoning"]
        + "\n"
        + "```sql\n"
        + example["reference_sql"]
        + "```"
        + "<|eot_id|>"
    )
