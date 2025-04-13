import pickle


with open("results/qwen/insertion_data_output.pickle", "rb") as f:
    result = pickle.load(f)
    
print(result)