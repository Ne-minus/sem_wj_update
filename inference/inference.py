import torch
import os
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from dataset import PromptDataset
from config import BATCH_SIZE, MODEL_NAME, INPUT_DATA_PATH, OUTPUT_DATA_PATH

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,4" 

def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_to_pickle(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Visible GPUs: {torch.cuda.device_count()}")
    model = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=2,  
        dtype="float16",        
        trust_remote_code=True,
        gpu_memory_utilization=0.8,  
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        skip_special_tokens=True
    )

    pickle_files = os.listdir(INPUT_DATA_PATH)

    for pickle_file in pickle_files:
        print(f"Processing file: {pickle_file}", flush=True)
        
        data = load_pickle_file(f"{INPUT_DATA_PATH}/{pickle_file}")
        
        dataset = PromptDataset(data, tokenizer)
        print(f"Dataset size: {len(dataset)}")
        
        batch_size = BATCH_SIZE
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        results = []

        for batch in tqdm(dataloader):
            input_texts = batch['input_text']
            targets = batch['target']
            
            outputs = model.generate(input_texts, sampling_params)
            
            for target, output in zip(targets, outputs):
                response = output.outputs[0].text
                results.append({
                    "target": target,
                    "output": response
                })

        output_file = f"{OUTPUT_DATA_PATH}/{pickle_file.replace('.pickle', '_output.pickle')}"
        save_to_pickle(output_file, results)
        
        print(f"Saved results to: {output_file}", flush=True)
        
if __name__ == "__main__":
    main()