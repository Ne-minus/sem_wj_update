from openai import OpenAI
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import PromptDataset
from config import BATCH_SIZE, MODEL_NAME, INPUT_DATA_PATH, OUTPUT_DATA_PATH

def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_to_pickle(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
        
def makedirect(path):
    if not os.path.exists(path):
        os.makedir(path)

def main():

    client = OpenAI(
        base_url="http://vllm-neminova:8000/v1",
        api_key="token-abc123"
    )
    print(os.listdir(INPUT_DATA_PATH))
    pickle_files = os.listdir(INPUT_DATA_PATH)
    
    makedirect(OUTPUT_DATA_PATH)

    for pickle_file in pickle_files:
        print(f"Processing file: {pickle_file}", flush=True)

        data = load_pickle_file(f"{INPUT_DATA_PATH}/{pickle_file}")
        dataset = PromptDataset(data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

        results = []

        for batch in tqdm(dataloader):
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=batch[0]
                )
            
            result = {"target": batch[1] , "result": completion.choices[0].message.content}
            
            results.append(result)

        output_file = f"{OUTPUT_DATA_PATH}/{pickle_file.replace('.pickle', '_output.pickle')}"
        save_to_pickle(output_file, results)
        print(f"Saved results to: {output_file}", flush=True)

if __name__ == "__main__":
    main()
