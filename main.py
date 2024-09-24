import argparse
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from pathlib import Path
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from consistency_models.utils import update_ema_model_
from consistency_models import ConsistencySamplingAndEditing, ImprovedConsistencyTraining, pseudo_huber_loss
from grm import GenerativeRepresentation
from unet.unet_config import unet_config, encoder_config
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='grm')
parser.add_argument('--ema', action='store_true', help='whether to use ema')
parser.add_argument('--detach', action='store_true', help='whether to detach output of generator')
parser.add_argument('--data', default='local_datasets/miniimagenet', type=Path, metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loader workers')
parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--img-size', default=84, type=int, help='size of input data')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--output-dir', default='./output/', type=Path, metavar='DIR', help='path to output directory')  
parser.add_argument('--resume', type=str, default=None, help='path to resume checkpoint')

args = parser.parse_args()

def main():
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/checkpoints').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/images').mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    
    # Create model
    cm_model = unet_config.make_model().to(device)
    encoder_model = GenerativeRepresentation(encoder_config.make_model(), 768, 3072).to(device)
    cm_model.train()

    use_ema = args.ema

    if use_ema:
        cm_model_ema = deepcopy(cm_model)
        cm_model_ema.load_state_dict(cm_model.state_dict())
        for param in cm_model_ema.parameters():
            param.requires_grad = False
        cm_model_ema.eval()

    # Load checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        cm_model.load_state_dict(checkpoint['cm_model_state_dict'])
        encoder_model.load_state_dict(checkpoint['encoder_model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
        if use_ema:
            cm_model_ema = cm_model_ema.load_state_dict(checkpoint['cm_model_ema_state_dict'])

    print("\n--- Training Configuration ---")
    print(f"Number of Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Use EMA: {use_ema}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"Output Directory: {args.output_dir}\n")

    print("--- Model Summary ---")
    cm_model_params = sum(p.numel() for p in cm_model.parameters()) / 1_000_000  # Convert to millions
    encoder_model_params = sum(p.numel() for p in encoder_model.parameters()) / 1_000_000  # Convert to millions
    print(f"CM Model Parameters: {cm_model_params:.2f} million")
    print(f"Encoder Model Parameters: {encoder_model_params:.2f} million")
    print("-----------------------\n")

    # Training configurations
    batch_size = args.batch_size
    num_epochs = args.epochs

    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(args.data, transform=transform)
    fid_dataset = ImageFolder(args.data, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    fid_loader = torch.utils.data.DataLoader(dataset=fid_dataset, batch_size=batch_size, shuffle=False)

    optimizer_backbone = torch.optim.RAdam(cm_model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_encoder = torch.optim.RAdam(encoder_model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    scheduler_backbone = torch.optim.lr_scheduler.LinearLR(optimizer_backbone, start_factor=1e-5, total_iters=1000)
    scheduler_encoder = torch.optim.lr_scheduler.LinearLR(optimizer_encoder, start_factor=1e-5, total_iters=1000)

    # Initialize the training module 
    improved_consistency_training = ImprovedConsistencyTraining(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        sigma_data=0.5,
        initial_timesteps=10,
        final_timesteps=1280,
        lognormal_mean=-1.1,
        lognormal_std=2.0,
    )
    
    consistency_sampling_and_editing = ConsistencySamplingAndEditing(
        sigma_min=0.002,
        sigma_data=0.5,
    )

    fid = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(device)

    for i, batch in enumerate(fid_loader):
        fid.update(batch[0].to(device), real=True)

    current_training_step = 0
    total_training_steps = len(train_loader)
    total_training_steps = num_epochs * len(train_loader)
    
    for epoch in range(start_epoch, num_epochs):
        for i, (images, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):

            images = images.to(device)
            optimizer_backbone.zero_grad()
            optimizer_encoder.zero_grad()

            # Forward Pass
            recon_output = improved_consistency_training(cm_model, images, current_training_step, total_training_steps)

            if use_ema:
                ema_recon = improved_consistency_training(cm_model_ema, images, current_training_step, total_training_steps)
                x1, x2 = ema_recon.predicted, ema_recon.target
            else:
                x1, x2 = recon_output.predicted, recon_output.target

            # Loss Computation
            recon_loss = (pseudo_huber_loss(x1, x2) * recon_output.loss_weights).mean()

            if args.detach:
                aug_1, aug_2 = x1.detach(), x2.detach()
            aug_1, aug_2 = x1, x2

            rep_loss = encoder_model(aug_1, aug_2)
    
            loss = recon_loss + rep_loss
            # Backward Pass & Weights Update
            loss.backward()
            optimizer_backbone.step()
            optimizer_encoder.step()

            scheduler_backbone.step()
            scheduler_encoder.step()

            if use_ema:
                update_ema_model_(cm_model_ema, cm_model, 0.99993)

            current_training_step += 1
            
            # Logging to TensorBoard
            if (current_training_step) % 10 == 0:
                writer.add_scalar('Loss/recon_loss', recon_loss.item(), current_training_step)
                writer.add_scalar('Loss/rep_loss', rep_loss.item(), current_training_step)
                writer.add_scalar('Training/epoch', epoch + 1, current_training_step)
                writer.add_scalar('num_timestep', recon_output.num_timesteps)

        # Sample and log images
        samples_one_step = consistency_sampling_and_editing(cm_model, torch.randn((64, 3, 32, 32), device=device), sigmas=[80.0], clip_denoised=True, verbose=True)
        save_image((samples_one_step / 2 + 0.5).cpu().detach(), f'{args.output_dir}/images/one_images_{epoch+1}.png')
        writer.add_image('Sample/one_step_online', (samples_one_step / 2 + 0.5).cpu().detach(), epoch + 1)

        samples_few_step = consistency_sampling_and_editing(cm_model, torch.randn((64, 3, 32, 32), device=device), sigmas=(80.0, 24.4, 5.84, 0.9, 0.661), clip_denoised=True, verbose=True)
        save_image((samples_few_step / 2 + 0.5).cpu().detach(), f'{args.output_dir}/images/few_images_{epoch+1}.png')
        writer.add_image('Sample/few_step_online', (samples_few_step / 2 + 0.5).cpu().detach(), epoch + 1)

        if use_ema:
            samples_one_step = consistency_sampling_and_editing(cm_model_ema, torch.randn((64, 3, 32, 32), device=device), sigmas=[80.0], clip_denoised=True, verbose=True)
            save_image((samples / 2 + 0.5).cpu().detach(), f'{args.output_dir}/images/one_ema_images_{epoch+1}.png')
            writer.add_image('Sample/one_step_ema', (samples_one_step / 2 + 0.5).cpu().detach(), epoch + 1)

            samples_few_step = consistency_sampling_and_editing(cm_model_ema, torch.randn((64, 3, 32, 32), device=device), sigmas=(80.0, 24.4, 5.84, 0.9, 0.661), clip_denoised=True, verbose=True)
            save_image((samples / 2 + 0.5).cpu().detach(), f'{args.output_dir}/images/few_ema_images_{epoch+1}.png')
            writer.add_image('Sample/few_step_ema', (samples_few_step / 2 + 0.5).cpu().detach(), epoch + 1)

        if (epoch + 1) % 10 == 0:
            for i in range(int(50000 / batch_size)):
                with torch.no_grad():
                    samples = consistency_sampling_and_editing(
                        cm_model_ema,
                        torch.randn((batch_size, 3, 32, 32), device=device),
                        sigmas=[80.0],
                        clip_denoised=True,
                        verbose=True)
                image = (samples / 2 + 0.5).clamp(0, 1)
                fid.update(image.cpu(), real=False)
            fid_result = float(fid.compute())
            print(f"Epoch [{epoch+1}], FID: {fid_result}")
            writer.add_scalar('FID', fid_result, epoch + 1)
            fid.reset()

        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'cm_model_state_dict': cm_model.state_dict(),
                'encoder_model_state_dict': encoder_model.state_dict(),
            }
            if use_ema:
                checkpoint['cm_model_ema_state_dict'] =  cm_model_ema.state_dict()

    torch.save(checkpoint, f'{args.output_dir}/checkpoints/ict_online_{epoch+1}.pth')
    torch.save(encoder_model.state_dict(), f'{args.output_dir}/checkpoints/grm_{epoch+1}.pth')
    if use_ema:
        torch.save(cm_model_ema.state_dict(), f'{args.output_dir}/checkpoints/ict_ema_{epoch+1}.pth')
    writer.close()

if __name__ == '__main__':
    main()