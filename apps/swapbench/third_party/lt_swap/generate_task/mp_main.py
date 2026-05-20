
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Imports
import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from re import S
from typing import List, Optional, Tuple
import pandas as pd

try:
    from matrix import Cli
    from matrix.client import query_llm
except ImportError:
    print("Warning: 'matrix' is not installed. Visit https://github.com/facebookresearch/matrix if you want to install it.")

try:
    #Import to use the official Python SDKs of 3P LLMs
    #Run 'pip install openai' in the CLI if openai is not installed
    import openai
    from openai import AsyncOpenAI
    from openai import AsyncAzureOpenAI
except ImportError:
    print("Warning: 'openai' is not installed.")

# Class for LMLClient

class LLMClient:
    """
    Abstract base class.
    """

    async def get_llm_response(self, prompt: str, temperature: float) -> str:
        raise NotImplementedError

class OpenAIClient(LLMClient): # Illustration, to customize

    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def get_llm_response(self, prompt: str, temperature: float) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

class AzureOpenAIClient(LLMClient): # Illustration, to customize

    def __init__(self, api_endpoint: str, api_key: str, model: str, api_version: str = "2024-10-21"):
        self.model = model
        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            api_key=api_key,
            azure_endpoint=api_endpoint
        )

    async def get_llm_response(self, prompt: str, temperature: float) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

class MatrixClient(LLMClient): # Illustration, to send request to a Matrix ray cluster (https://github.com/facebookresearch/matrix)

    def __init__(self, app_name: str):            
        self.app_name = app_name
        self.metadata = Cli().get_app_metadata(app_name=self.app_name)

    async def get_llm_response(self, prompt: str, temperature: float) -> str:
        response = await query_llm.make_request(
            url=self.metadata["endpoints"]["head"],
            model=self.metadata["model_name"],
            app_name=self.metadata["name"],
            data={"messages": [{"role": "user", "content": prompt}]},
            temperature=temperature
        )

        return "\n".join(response['response']['text']).strip()

class TestClient(LLMClient):
    """
    A dummy LLMClient that echoes the prompt back as the response.
    Simulates real-world behavior with random failures and processing delays.
    """

    async def get_llm_response(self, prompt: str, temperature: float) -> str:
        import asyncio
        import random

        # Simulate processing time (between 0.05 and 0.2 seconds)
        processing_time = random.uniform(0.05, 0.2)
        await asyncio.sleep(processing_time)

        # Randomly fail with 1% probability
        if random.random() < 0.01:
            raise Exception("Simulated random failure in TestClient")

        # Sometimes return a slightly modified response to simulate LLM variations
        if random.random() < 0.3:
            variations = [
                f"Processed prompt: {prompt}",
                f"Your input was: {prompt}",
                f"TestClient received: {prompt}",
            ]
            return random.choice(variations)

        # Default response
        return f"The input prompt was: {prompt}"


class TestClientFiltering(LLMClient):
    """
    A dummy LLMClient that gives a random answer between A and B.
    Simulates real-world behavior with random failures and processing delays.
    """

    async def get_llm_response(self, prompt: str, temperature: float) -> str:
        import asyncio
        import random

        # Simulate processing time (between 0.05 and 0.2 seconds)
        processing_time = random.uniform(0.05, 0.2)
        await asyncio.sleep(processing_time)

        # Randomly fail with 1% probability
        if random.random() < 0.01:
            raise Exception("Simulated random failure in TestClient")

        # Pick at random between A and B
        if random.random() < 0.5:
            return "[A]"
        else:
            return "[B]"


# UTILS

async def with_retries(func, *args, max_retries=3, **kwargs):
    """
    Retry a function with exponential backoff.
    """
    for i in range(max_retries):
        try:
            return await asyncio.wait_for(func(*args, **kwargs), 180) # timeout worker after 3 m
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Error: {str(e)[:50]}")
            if i < max_retries - 1:
                print(f"Retrying in {2 ** (i + 1)} seconds...")
                await asyncio.sleep(2 ** (i + 1))
            else:
                raise


# WORKER AND QUEUE

async def worker(
    name: int,
    queue: asyncio.Queue,
    client: LLMClient,
    lock: asyncio.Lock,
    output_file,
    max_retries: int,
    temperature: float,
):
    """
    Worker function that consumes items from the queue and processes them.
    """
    processed_count = 0
    while True:
        item = await queue.get()
        if item is None:
            print(f"Worker {name} finished after processing {processed_count} items")
            queue.task_done()  # Signal to the queue that this worker is done
            return
        idx, line, prompt, answer = item
        try:
            if processed_count % 100:
                pass
                #print(f"Worker {name} processed items: {processed_count}")
            response = await with_retries(
                client.get_llm_response,
                prompt.replace("\\n", "\n"),
                temperature=temperature,
                max_retries=max_retries,
            )
            processed_count += 1
        except Exception as e:
            response = f"<ERROR after {max_retries} retries: {str(e)[:200]}>"  # Limiting the size of the error message recorded in the file
            print(
                f"Worker {name} failed to process item {idx} after {max_retries} retries"
            )

        # Write the result to the output file, keeping the original metadata as well as idx of the input line (in case need to realign)
        # Using a lock to ensure that only one worker can write to the file at a time
        response = (
            response.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        )
        async with lock:
            if answer is None:
                output_file.write(f"{idx}|{line}|{response}\n")
            else:
                output_file.write(f"{idx}|{line}|{answer}|{response}\n")
            output_file.flush()

        queue.task_done()


async def main(args):
    """
    Main function that creates a queue, starts workers, and feeds the queue with items.
    List of arguments: client, api_key, model, temperature, input_file, output_file, num_workers, queue_size, max_retries
    """
    print(f"Starting processing with {args.num_workers} workers")

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting processing at {start_datetime} with {args.num_workers} workers")

    # Instanciate the LLM Client
    if args.client == "openai":
        if not args.api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print(
                    "Please set the OPENAI_API_KEY environment variable, or provide it as an argument.", file=sys.stderr
                )
                sys.exit(1)
        else:
            api_key = args.api_key
        client = OpenAIClient(api_key, model=args.model)
        print(f"Using OpenAI client with model: {args.model}")
    elif args.client == "azureopenai":
        if not args.api_key or not args.api_endpoint:
            print(
                "Please make sure to provide AzureOpenAI api_key and api_endpoint as arguments", file=sys.stderr
            )
            sys.exit(1)
        else:
            api_key = args.api_key
            api_endpoint = args.api_endpoint
        client = AzureOpenAIClient(api_endpoint, api_key, model=args.model)
        print(f"Using AzureOpenAI client at {api_endpoint} with model: {args.model}")
    elif args.client == "test":
        client = TestClient()
        print("Using Test client")
    elif args.client == "test_filtering":
        client = TestClientFiltering()
        print("Using Test Filtering client")
    elif args.client == "matrix":
        client = MatrixClient(app_name=args.app_name)
        print("Using Matrix client")
    else:
        print(f"Unknown client: {args.client}", file=sys.stderr)
        sys.exit(1)

    # Create a queue to store the items
    queue = asyncio.Queue(maxsize=args.queue_size)

    # Create a lock to ensure that only one worker can write to the file at a time
    lock = asyncio.Lock()

    # Open the output file for writing
    # If the file already exists, append to it; to handle resuming
    mode = "a" if os.path.exists(args.output_file) else "w"
    print(mode)
    output_file = open(args.output_file, mode, encoding="utf-8")
    done_indices = set()
    error_indices = 0
    if mode == "a":
        # Read the existing file and get the indices of the lines that have already been processed
        print("Output file exists, checking for already processed items...")
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    idx = str(
                        line.split("|", 1)[0]
                    )  # Only works if the first column is the index
                except ValueError:
                    # Consider not processed is malformed/partial
                    continue
                try:
                    if not line.rsplit("|", 1)[1].startswith("<ERROR"):
                        done_indices.add(str(idx))
                    else:
                        error_indices += 1
                except:
                    continue
        print(
            f"Found {len(done_indices)} already processed items, and {error_indices} lines with <ERROR in the output file which will be retried."
        )

    # Import the worker function here to avoid unbound reference
    from __main__ import worker

    # Create the workers
    print(f"Creating {args.num_workers} worker tasks")
    workers = [
        asyncio.create_task(
            worker(
                i, queue, client, lock, output_file, args.max_retries, args.temperature
            )
        )
        for i in range(args.num_workers)
    ]

    # Feed the queue with items from input file
    print(f"Reading input file: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()
        total_lines = len(lines)
        print(f"Found {total_lines} lines in input file")
        items_to_process = 0

        for i, line in enumerate(lines):
            # Split the line into prompt and response
            line, prompts = line.rsplit("|", 1)

            # Prompt may actually contain several prompts, separated by a /
            # We will process each prompt separately

            # But first we count how many "/" appear, it should be an even number as the first half are the prompts and the second are the answers
            slash_count = prompts.count("/")
            if slash_count == 0:  # typical case, one prompt, no answer to process
                idx = f"{i}"  # Create an index that reflect the line number for input file as well as the number of the nested prompt in case of multiple prompt per line in input file
                if idx in done_indices:
                    # Skip lines that have already been processed
                    continue

                # Add the item to the queue
                item = (idx, line, prompts.strip(), None)
                if not item[2]:  # Skip empty prompts
                    continue
                await queue.put(item)
                items_to_process += 1

                # Print progress every 100 items
                if items_to_process % 100 == 0:
                    print(f"Queued {items_to_process} items for processing")

            elif slash_count % 2 != 1:
                print(
                    f"Warning: Line {i} has an even number of slashes ({slash_count}) and has been skipped."
                )  # issue
                continue

            else:
                for j, prompt_answer in enumerate(
                    zip(
                        prompts.split("/")[: slash_count // 2 + 1],
                        prompts.split("/")[slash_count // 2 + 1 :],
                    )
                ):  # Loop through each prompt and associated answer
                    prompt, answer = prompt_answer
                    idx = f"{i}-{j}"  # Create an index that reflect the line number for input file as well as the number of the nested prompt in case of multiple prompt per line in input file
                    if idx in done_indices:
                        # Skip lines that have already been processed
                        continue

                    # Add the item to the queue
                    item = (
                        idx,
                        line,
                        prompt.strip(),
                        answer.strip(),
                    )  # We avoid duplicating line content
                    if not item[2]:  # Skip empty prompts
                        continue
                    await queue.put(item)
                    items_to_process += 1

                    # Print progress every 100 items
                    if items_to_process % 100 == 0:
                        print(f"Queued {items_to_process} items for processing")

        print(
            f"Queued {items_to_process} items for processing out of {total_lines} total lines"
        )

    # Signal to the workers that there are no more items to process
    print("All items queued, sending termination signals to workers")
    for _ in workers:
        await queue.put(None)

    # Wait for all workers to finish
    print("Waiting for all workers to complete...")
    await queue.join()
    print("All items have been processed")

    # Cancel worker tasks
    for worker in workers:
        worker.cancel()

    await asyncio.gather(*workers, return_exceptions=True)

    # Close the output file
    output_file.close()

    # Calculate and print execution time
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = execution_time % 60

    # Print a message when all workers have finished
    print(f"All workers have finished processing the items at {end_datetime}.")
    print(
        f"Total execution time: {hours}h {minutes}m {seconds:.2f}s ({execution_time:.2f} seconds)"
    )