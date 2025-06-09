"""
Variational Autoencoder (VAE) for 64×64 RGB images
=================================================
• Architecture: 4‑level strided Conv encoder ↔ ConvTranspose decoder.
• Latent dimension, learning rate, etc. are CLI flags.
• Training loop saves checkpoints + sample grids every 10 epochs.

Usage (single‑GPU):
$ python vae64_pytorch.py --data_root /path/to/images --epochs 100 --latent_dim 256

Data format: ImageFolder‑style directory with class sub‑dirs; images will be resized to 64×64.
"""
import os, argparse, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import wandb
from diffusers.optimization import get_scheduler
from dsprites import *

# ---------- Model Blocks ----------
class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 → 8
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8 → 4
            nn.BatchNorm2d(256), nn.ReLU(True),
        )
        feat_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 → 8
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 → 16
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16 → 32
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 32 → 64
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(z)

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder, self.decoder = Encoder(latent_dim), Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# ---------- Loss & Train ----------
@torch.no_grad()
def sample_grid(decoder, latent_dim, device):
    z = torch.randn(64, latent_dim, device=device)
    grid = decoder(z).cpu()
    # utils.save_image(grid, f"samples/samples_{epoch:03d}.png", nrow=8)
    return grid

def vae_loss(x_hat, x, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld, recon, kld

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", help="ImageFolder root dir")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
        )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
        )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )

    return parser

def main():
    args = get_arguments().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
    train_transforms = transforms.Compose(
        [
            transforms.Resize(64, interpolation=interpolation),  # Use dynamic interpolation method
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = prepare_dsprites()
    dataset = dataset['train'].train_test_split(0.1)

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples


    train_dataset = dataset['train'].with_transform(preprocess_train)
    test_dataset = dataset['test'].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}
    

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=1,
        drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=1,
        drop_last=True
    )

    model = VAE(args.latent_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.adam_weight_decay,
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps_for_scheduler = num_update_steps_per_epoch * args.epochs

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_training_steps_for_scheduler,
    )
    
    epochs = args.epochs

    wandb.init(project="dsprites_vae")
    global_step = 0
    num_val_samples = 50

    for epoch in range(epochs):
        model.train()
        total = 0
        for batch in train_dataloader:
            
            batch['pixel_values'] = batch['pixel_values'].to(device)
            x_hat, mu, logvar = model(batch['pixel_values'])
            loss, _, _ = vae_loss(x_hat, batch['pixel_values'], mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total += loss.item()

            wandb.log({"train/loss": loss.item(), "step": global_step})
            global_step += 1
        
        test_loss_total = 0
        model.eval()
        for batch in test_dataloader:
            batch['pixel_values'] = batch['pixel_values'].to(device)
            x_hat, mu, logvar = model(batch['pixel_values'])
            loss, _, _ = vae_loss(x_hat, batch['pixel_values'], mu, logvar)
            test_loss_total += loss.item()

        wandb.log({"test/avg loss": test_loss_total / len(test_dataloader)})

        val_images = []
        for s in range(num_val_samples):
            img = sample_grid(model.decoder, args.latent_dim, device)
            val_images.append(img)
        

        wandb.log(
            {
                "validation": [
                    wandb.Image(image, caption=f"{i}")
                    for i, image in enumerate(val_images)
                ]
            }
        )


def train_epoch(model, loader, opt, device):
    model.train(); total = 0
    for x, _ in loader:
        x = x.to(device)
        opt.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, _, _ = vae_loss(x_hat, x, mu, logvar)
        loss.backward(); opt.step()
        total += loss.item()
    return total / len(loader.dataset)



# ---------- CLI ----------
if __name__ == "__main__":
    main()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.makedirs("samples", exist_ok=True)

    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)), transforms.ToTensor()])
    # ds = datasets.ImageFolder(args.data_root, transform)
    # dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # model = VAE(args.latent_dim).to(device)
    # opt = optim.Adam(model.parameters(), lr=args.lr)

    # for epoch in range(1, args.epochs + 1):
    #     loss = train_epoch(model, dl, opt, device)
    #     print(f"epoch {epoch} | loss {loss:.2f}")
    #     if epoch % 10 == 0:
    #         sample_grid(model.decoder, args.latent_dim, epoch, device)
    # torch.save(model.state_dict(), "vae64.pth")
