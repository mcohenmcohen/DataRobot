
labels = pd.read_csv('../data/labels.csv', header=None)

# Tokenize input
ids_muilt_seq = tokenizer(data, truncation=True, padding=True, return_tensors="pt") # tokenizer returns the ids

# Get the logits
outputs = model_pixability(**ids_muilt_seq)

# Get the probabilties
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Convert to numpy, then dataframe
pred_np = predictions.detach().numpy()

pd.DataFrame(pred_np , columns=labels[0].values)
