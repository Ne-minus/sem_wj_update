from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_context = self.data[idx][0]
        target = self.data[idx][1]

        system_part = "You are a helpful assistant. List all the possible words divided with a comma"
        

        messages = [
            {"role": "system", "content": system_part},
            {"role": "user", "content": user_context}
        ]

        return messages, target