import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import wandb
import os
from transformers.optimization import get_scheduler
from dsprites import *

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool   = nn.MaxPool2d(2)

        # Dropouts
        self.drop2d = nn.Dropout2d(p=0.25)   # zero-out whole channels
        self.drop   = nn.Dropout(p=0.2)      # zero-out individual neurons

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop2d(x)                   # after first block

        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2d(x)                   # after second block

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)                     # before final classifier
        return self.fc2(x)

def main():
    device = "cuda"
    dataset = prepare_dsprites()

    train_transforms = transforms.Compose(
        [
            transforms.Resize(64),  # Use dynamic interpolation method
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        )

    def preprocess_train(examples):
        # images = [image.convert("RGB") for image in examples[image_column]]
        images = [image for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        labels = torch.stack([torch.tensor(example["label_shape"]) for example in examples])
        return {"pixel_values": pixel_values, "label": labels}

    dataset = dataset['train'].train_test_split(0.1)
    train_dataset = dataset['train'].with_transform(preprocess_train)
    test_dataset = dataset['test'].with_transform(preprocess_train)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=2048,
    )

    print(f"There are {len(train_dataloader)} steps per epoch")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=512,
    )

    epochs = 150

    num_training_steps = (len(train_dataloader) * epochs)
    num_warmup_steps = int(0.1 * num_training_steps)
    model = SimpleCNN().to(device)
    opt = torch.optim.AdamW(params = model.parameters(), lr=1e-5)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=opt,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    criterion = nn.CrossEntropyLoss()

    

    wandb.init(project="dsprites_classifier")
    global_step = 0

    output_dir = "dsprites_classifier_dropout_02_new"
    os.makedirs(output_dir, exist_ok=True)

    best_acc=0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for i, batch in enumerate(train_dataloader):
            imgs, labels = batch['pixel_values'].to(device), batch['label'].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            train_loss += loss.item()
            
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            wandb.log({"train/loss": loss.item(), "train/step": global_step, "train/step_acc": preds.eq(labels).sum().item() / labels.size(0)})
            global_step += 1


        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                imgs, labels = batch['pixel_values'].to(device), batch['label'].to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_loss += loss
                _, preds = outputs.max(1)
                test_correct += preds.eq(labels).sum().item()
                test_total += labels.size(0)


            test_acc = test_correct/test_total
            wandb.log({"test/avg_loss": test_loss / len(test_dataloader), "test/acc": test_acc})

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"{output_dir}/checkpoint_{global_step}.pth")

if __name__ == "__main__":
    main()