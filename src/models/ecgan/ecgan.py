import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class ECGAN:
    def __init__(self, input_size, window_size, lr=1e-4, ssl_lr=1e-3, name='ECGAN'):
        """
        Initialize the ECGAN model with generator, discriminator, and SSL components.

        Args:
            input_size (int): Size of the input time-series data.
            window_size (int): Length of the time-series window.
            lr (float): Learning rate for the generator and discriminator.
            ssl_lr (float): Learning rate for the self-supervised learning (SSL) component.
            name (str): Name of the model instance.
        """
        self.input_size = input_size
        self.window_size = window_size
        self.name = name

        # Initialize models
        self.ssl = self._build_ssl()
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Optimizers
        self.ssl_optimizer = optim.Adam(self.ssl.parameters(), lr=ssl_lr)
        self.generator_optimizer = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.discriminator_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=lr)

        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # Used for discriminator and generator
        self.reconstruction_loss = nn.MSELoss()  # Used for SSL reconstruction

        # History to track training losses
        self.history = {'generator': [], 'discriminator': [], 'ssl': []}

    def _build_ssl(self):
        """
        Build the self-supervised learning (SSL) autoencoder model.

        Returns:
            nn.Module: The SSL model.
        """
        return nn.Sequential(
            nn.LSTM(input_size=1, hidden_size=128, num_layers=6, batch_first=True),
            nn.Linear(128, 10),
            nn.Sigmoid()
        )

    def _build_generator(self):
        """
        Build the generator model.

        Returns:
            nn.Module: The generator model.
        """
        return nn.Sequential(
            nn.LSTM(input_size=10, hidden_size=128, num_layers=2, batch_first=True),
            nn.Linear(128, self.window_size),
            nn.Tanh()
        )

    def _build_discriminator(self):
        """
        Build the discriminator model.

        Returns:
            nn.Module: The discriminator model.
        """
        return nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def train_ssl(self, real_data, epochs=10):
        """
        Train the self-supervised learning (SSL) component.

        Args:
            real_data (torch.Tensor): Input time-series data.
            epochs (int): Number of training epochs.
        """
        self.ssl.train()
        for epoch in range(epochs):
            self.ssl_optimizer.zero_grad()

            # Forward pass: obtain latent representation and reconstruct data
            latent_repr, _ = self.ssl(real_data)
            reconstructed = self.generator(latent_repr)

            # Compute reconstruction loss
            loss = self.reconstruction_loss(reconstructed, real_data)

            # Backpropagation and optimization
            loss.backward()
            self.ssl_optimizer.step()

            print(f"[SSL] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            self.history['ssl'].append(loss.item())

    def train_adversarial(self, real_data):
        """
        Perform adversarial training for one step.

        Args:
            real_data (torch.Tensor): Real input data.

        Returns:
            tuple: Generator loss and discriminator loss.
        """
        # Generate fake data
        latent_repr, _ = self.ssl(real_data)
        fake_data = self.generator(latent_repr)

        # Train discriminator
        self.discriminator_optimizer.zero_grad()
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data.detach())

        real_loss = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Train generator
        self.generator_optimizer.zero_grad()
        fake_pred = self.discriminator(fake_data)
        generator_loss = self.adversarial_loss(fake_pred, torch.ones_like(fake_pred))
        generator_loss.backward()
        self.generator_optimizer.step()

        return generator_loss.item(), discriminator_loss.item()

    def fit(self, dataset, ssl_epochs, adv_epochs, batch_size):
        """
        Training phase ECGAN with the provided dataset.

        Args:
            dataset (Dataset): Input dataset for training.
            ssl_epochs (int): Number of epochs for self-supervised pretraining.
            adv_epochs (int): Number of epochs for adversarial training.
            batch_size (int): Size of training batches.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train SSL first (pretraining)
        print(f"Training SSL phase for {ssl_epochs} epochs...")
        for epoch in range(ssl_epochs):
            for real_data in dataloader:
                real_data = real_data.to(torch.float32)
                self.train_ssl(real_data, epochs=1)

        # Then train adversarial phase
        print(f"Training adversarial phase for {adv_epochs} epochs...")
        for epoch in range(adv_epochs):
            for real_data in dataloader:
                real_data = real_data.to(torch.float32)
                gen_loss, disc_loss = self.train_adversarial(real_data)

                self.history['generator'].append(gen_loss)
                self.history['discriminator'].append(disc_loss)

            print(f"Epoch {epoch+1}/{adv_epochs}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")
