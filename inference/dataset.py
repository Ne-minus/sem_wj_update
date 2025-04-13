from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_context = self.data[idx][0]
        target = self.data[idx][1]
        
        system_part = "You are a helpful assistant. List all the possible words divided with a coma. Your answer should not include anything except the words divided by a coma."
        
        messages = [
            {"role": "system", "content": system_part},
            {"role": "user", "content": user_context}
        ]
        
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        encodings = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'target': target,
            'input_text': input_text  # We'll need this for vLLM
        }