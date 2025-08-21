import os, argparse, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import wandb
from diffusers.optimization import get_scheduler
from dsprites import *
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, data_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.GELU()
            )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_std = nn.Linear(hidden_dim, latent_dim)
        # decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        enc_x = self.encoder_fc(x)
        mu = self.encoder_mu(enc_x)
        sigma = self.encoder_std(enc_x)
        return mu, sigma
    
    def reparameterize(self, mu, sigma):
        std = (0.5 * sigma).exp()
        return mu + std * torch.randn_like(std)
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decoder(z), mu, sigma

class ConvVAE(nn.Module):
    """
    64×64×1 → latent → 64×64×1
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()

        # ────────────── encoder ──────────────
        self.enc = nn.Sequential(                     # out‑shape
            nn.Conv2d(1,   32, 4, 2, 1), nn.GELU(),   # 32×32×32
            nn.Conv2d(32,  64, 4, 2, 1), nn.GELU(),   # 64×16×16
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),   # 128×8×8
            nn.Conv2d(128,256, 4, 2, 1), nn.GELU()    # 256×4×4
        )
        self.flatten     = nn.Flatten()               # 256·4·4 = 4096
        self.fc_mu       = nn.Linear(4096, latent_dim)
        self.fc_logvar   = nn.Linear(4096, latent_dim)

        # ────────────── decoder ──────────────
        self.fc_dec = nn.Linear(latent_dim, 4096)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.GELU(),  # 128×8×8
            nn.ConvTranspose2d(128, 64,4,2,1), nn.GELU(),  # 64×16×16
            nn.ConvTranspose2d( 64, 32,4,2,1), nn.GELU(),  # 32×32×32
            nn.ConvTranspose2d( 32,  1,4,2,1), nn.Sigmoid()# 1×64×64
        )

    # ────────────── helpers ──────────────
    def encode(self, x):
        h = self.flatten(self.enc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.dec(h)

    # ────────────── forward ──────────────
    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

# ---------- Loss & Train ----------
@torch.no_grad()
def sample_grid(model, latent_dim, device):
    z = torch.randn(1, latent_dim, device=device)
    grid = model.decode(z).cpu()
    # utils.save_image(grid, f"samples/samples_{epoch:03d}.png", nrow=8)
    return grid

# def vae_loss(x_hat, x, mu, sigma, beta):
#     recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="mean")
#     kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
#     # kld = 
#     return recon + beta * kld, recon, kld

def vae_loss(x_hat, x, mu, logvar):
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")

    # KL divergence (mean across batch)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())     # shape [B, D]
    kld = kld.sum(dim=1).sum()                               # scalar mean

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
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = prepare_dsprites()
    dataset = dataset['train'].train_test_split(0.1)

    def preprocess_train(examples):
        # images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in examples['image']]
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

    model = ConvVAE(args.latent_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
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
            # print(batch['pixel_values'].shape)
            # print(batch['pixel_values'].view(-1, 4096).shape)
            
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
            img = sample_grid(model, args.latent_dim, device)
            val_images.append(img)
        

        wandb.log(
            {
                "validation": [
                    wandb.Image(image.view(1, 64,64), caption=f"{i}")
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
