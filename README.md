# MyWeb
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# 定義一個簡單的資料集
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

# 創建一個小型的訓練資料集
texts = ["這是一個正面的例子", "這是一個負面的例子", "這是另一個正面的例子", "這是另一個負面的例子"]
labels = [1, 0, 1, 0]

train_dataset = CustomDataset(texts, labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 初始化BertTokenizer和BertForSequenceClassification模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 定義訓練參數
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

# 訓練模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
        labels = torch.tensor(batch['label']).unsqueeze(0).to(device)

        inputs.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 保存訓練好的模型
model.save_pretrained('your_model_directory')

```
