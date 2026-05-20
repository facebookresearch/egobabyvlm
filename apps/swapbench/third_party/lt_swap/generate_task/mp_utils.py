
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Post-processing to get a clean output file depending on if generation or filtering task

# Processing for filtering task

def format_answer(g):
    start = g.rfind("[")
    end = g.rfind("]")
    if start == -1 or end == -1:
        if g in ["A", "B"]:
            return g
        else:
            return None
    g = g[start + 1 : end]
    g = g.replace(" ", "")
    return g.upper()


def process_filtering_file(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    output_lines = []
    grouped_lines = {}

    # Group lines by the {i} part of idx
    for line in lines:
        idx = str(line.split("|", 1)[0])
        original_line, ground_truth, response = line.rsplit("|", 2)
        
        original_line = original_line.split("|", 1)[1]
        # Remove first part before |, this ensures original_line does not contain idx that the code added
        
        i, j = idx.split("-")
        if i not in grouped_lines:
            grouped_lines[i] = []
        grouped_lines[i].append((j, original_line, ground_truth, response.strip()))

    # Process each group
    for idx_i, group in grouped_lines.items():
        all_match = True
        responses = []
        store_line = None

        for j, original_line, ground_truth, response in sorted(
            group, key=lambda x: x[0]
        ):
            if format_answer(response) != ground_truth:
                all_match = False
                break
            responses.append(response)
            if store_line is None:
                store_line = original_line

        if all_match:
            responses_str = "|".join(responses)
            output_lines.append(f"{store_line}\n")

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(output_lines)

    print(f"Processed {input_file} and saved to {output_file}.")


# Processing for generation task
# Removing the first column (idx) from the output file of the LLM processing

def remove_file_idx(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f_in, open(
        output_file, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                # Remove the first column (everything before the first '|')
                parts = line.split("|", 1)
                if len(parts) > 1:
                    f_out.write(parts[1] + "\n")
                else:
                    f_out.write(line + "\n")  # In case there's no '|' character

    print(
        f"Processed {input_file} and saved to {output_file} with first column (idx) removed."
    )


# Overall function


def post_processing_file(input_file: str, output_file: str, task_type: str):
    if task_type == "generation":
        remove_file_idx(input_file, output_file)
    elif task_type == "filtering":
        process_filtering_file(input_file, output_file)
    else:
        raise (ValueError)

#### Creation of batches prior to LLM prompting

import os
import shutil
import math

def split_file(file_path: str, n: int) -> List[str]:
    # Create output directory if it doesn't exist
    output_dir = f"{file_path}_batches"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the base file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Calculate number of parts needed
    total_lines = len(lines)
    num_parts = math.ceil(total_lines / n)
    
    batch_files = []
    for i in range(num_parts):
        start_line = i * n
        end_line = start_line + n
        batch_lines = lines[start_line:end_line]
        
        # Write each batch to a separate file
        batch_file = os.path.join(output_dir, f"batch_{i}.txt")
        with open(batch_file, "w", encoding="utf-8") as bf:
            bf.writelines(batch_lines)
        
        batch_files.append(batch_file)
    
    return batch_files

def get_existing_batches(file_path: str, n: int) -> List[str]:
    # Generate the directory path based on file_path
    output_dir = f"{file_path}_batches"
    batch_files = []
    if os.path.exists(output_dir):
        for i in range(len(os.listdir(output_dir))):
            batch_file = os.path.join(output_dir, f"batch_{i}.txt")
            if os.path.exists(batch_file):
                batch_files.append(batch_file)
    return batch_files


##### Simple wrapper to send prompts to LLM

import glob
from pathlib import Path

# Version which allows early peaking
async def send_prompts_to_llm(task_type, input_file, batch_size=10_000, temperature=0, num_workers=50, queue_size=1_000, max_retries=3, app_name="", client="matrix", model="", early_stop=False, api_key="", api_endpoint=""):

    # Create or retrieve batch files
    batch_files = get_existing_batches(input_file, batch_size) or split_file(input_file, batch_size)

    assert task_type in ('generation', 'filtering')
    assert (temperature >=0) and (temperature <=1)
    
    example_args = argparse.Namespace(
        client=client,
        model=model,
        app_name=app_name,
        temperature=temperature,
        num_workers=num_workers,
        queue_size=queue_size,
        max_retries=max_retries,
        api_key=api_key,
        api_endpoint=api_endpoint
    )

    if early_stop:
        # Only concatenate files that end with _processed.txt
        
        final_output_file = os.path.join(f"{input_file}_final_output_early_stop.txt")
        
        with open(final_output_file, "wb") as final_out:
            print(input_file)
            for batch_processed_file in glob.glob(input_file+'_batches/*_processed.txt'):
                print(str(Path(input_file).parent)+'/*_processed.txt')
                print(batch_processed_file)
                batch_output_file = batch_processed_file.replace("_processed.txt", "_output_early_stop.txt")
        
                # Post process each processed batch file
                post_processing_file(batch_processed_file, batch_output_file, task_type)
        
                # Concatenate to final output file
                with open(batch_output_file, "rb") as infile:
                    shutil.copyfileobj(infile, final_out)

    else: #typical
        
        for batch_file in batch_files:
            # Generate corresponding output file names
            batch_processed_file = batch_file.replace(".txt", "_processed.txt")
        
            # Set input and processed file for each batch iteration
            example_args.input_file = batch_file
            example_args.output_file = batch_processed_file
        
            # Execute processing for each batch
            await main(example_args)
        
            # Display file size after processing
            print(
                f"Processed file {batch_processed_file} size: {os.path.getsize(batch_processed_file)} bytes"
            )
        
        # Final output concatenation
        final_output_file = os.path.join(f"{input_file}_final_output.txt")
        
        with open(final_output_file, "wb") as final_out:
            for batch_file in batch_files:
                batch_processed_file = batch_file.replace(".txt", "_processed.txt")
                batch_output_file = batch_file.replace(".txt", "_output.txt")
        
                # Post process each processed batch file
                post_processing_file(batch_processed_file, batch_output_file, task_type)
        
                # Concatenate to final output file
                with open(batch_output_file, "rb") as infile:
                    shutil.copyfileobj(infile, final_out)
        
    # Display final output file size
    print(f"Final output file size: {os.path.getsize(final_output_file)} bytes")