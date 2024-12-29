import os
import torch
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from model import GPT, GPTConfig, SequenceDDPMPipeline

# Training Pipeline
def train_pipeline(dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Training pipeline for the GPT-based model using a diffusion process.
    
    Args:
        dataloader: DataLoader object providing training data.
        device: Device to use for training ('cuda' or 'cpu').
        
    Returns:
        model: Trained GPT model.
        noise_scheduler: Configured noise scheduler.
    """

    # Constants and hyperparameters
    DIFFUSION_TIMESTEPS = 50       # Number of timesteps in the diffusion process
    BATCHSIZE = 64                # Batch size for training
    beta_start = 0.0001           # Start value for the beta schedule
    beta_end = 0.2                # End value for the beta schedule
    clip_sample = True            # Whether to clip sample values during the noise addition
    channels = 1                  # Number of channels in the input data
    epochs = 10                   # Number of training epochs
    checkpoint_dir = './checkpoints/'  # Directory for saving model checkpoints
    checkpoint_name = "jp_vowels.pt"   # File name for saving model weights

    # Initialize the noise scheduler for diffusion
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=DIFFUSION_TIMESTEPS, beta_end=beta_end, clip_sample=clip_sample
    )

    # Configure the GPT model
    config = GPTConfig(dim=channels, n_embd=128, n_head=4, n_layer=2, dropout=0.0)
    model = GPT(config).to(device)

    # Set up the optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=1e-2, betas=[0.9, 0.999])
    num_steps = len(dataloader) * epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=30, num_training_steps=num_steps
    )

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            optimizer.zero_grad()  # Reset gradients

            # Move batch to device and generate random timestep indices
            batch = batch.to(device)
            batch_size = batch.size(0)
            t = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), dtype=torch.long, device=device)

            # Add noise to the batch and predict the noise
            noise = torch.randn(batch.shape, device=device)
            noisy_sequence = noise_scheduler.add_noise(batch, noise, t)
            noise_pred = model(noisy_sequence, t)

            # Compute loss and backpropagate
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            # Update model weights and learning rate
            optimizer.step()
            lr_scheduler.step()

            global_step += 1

            # Save checkpoint periodically
            if (global_step + 1) % 100 == 0:
                print(f"Step {global_step + 1} - LR: {optimizer.param_groups[0]['lr']:.6f} - Loss: {loss.item()}")
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, checkpoint_name))

    print("Training Complete.")
    return model, noise_scheduler


# Inference Pipeline
def inference_pipeline(model, noise_scheduler, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Inference pipeline to generate samples using the trained model and diffusion process.
    
    Args:
        model: Trained GPT model.
        noise_scheduler: Configured noise scheduler.
        dataloader: DataLoader object providing input data for inference.
        device: Device to use for inference ('cuda' or 'cpu').
        
    Returns:
        samples: Generated samples from the model.
    """

    # Constants
    sample_batchsize = 128        # Batch size for sampling
    DIFFUSION_TIMESTEPS = 50      # Number of timesteps in the diffusion process

    model.eval()  # Set model to evaluation mode

    # Create the diffusion pipeline with the trained model and scheduler
    pipeline = SequenceDDPMPipeline(model, noise_scheduler).to(device)

    # Iterate over the dataloader to generate samples
    for batch in dataloader:
        pipe_output = pipeline(
            sample_batchsize, 
            num_inference_steps=DIFFUSION_TIMESTEPS, 
            generator=None, 
            return_all=True
        )
        samples = pipe_output.sequences
        return samples
