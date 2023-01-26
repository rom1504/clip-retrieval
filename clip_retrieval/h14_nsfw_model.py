"""Modeling & Loading code for H14 NSFW Detector"""

import os

import torch
from torch import nn


# pylint: disable=invalid-name
class H14_NSFW_Detector(nn.Module):
    """An NSFW detector for H14 CLIP embeds"""

    def __init__(self, input_size=1024, cache_folder=os.path.expanduser("~/.cache/clip_retrieval")):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, 1),
        )

        # Load the model from the cache folder
        self.load_state_dict(self.load_state(cache_folder))
        self.eval()

    def forward(self, x):
        """Forward pass of the model"""
        return self.layers(x)

    # pylint: disable=unused-argument
    def predict(self, x, batch_size):
        """autokeras interface"""
        with torch.no_grad():
            x = torch.from_numpy(x)
            y = self.layers(x)
            return y.detach().cpu().numpy()

    def load_state(self, cache_folder: str):
        """
        Load the model from the cache folder
        If it does not exist, create it
        """

        cache_subfolder = os.path.join(cache_folder, "h14_nsfw_model")
        if not os.path.exists(cache_subfolder):
            os.makedirs(cache_subfolder)

        model_path = os.path.join(cache_subfolder, "model.pt")
        if not os.path.exists(model_path):
            print("Downloading model...")
            import urllib.request  # pylint: disable=import-outside-toplevel

            urllib.request.urlretrieve(
                "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/raw/main/h14_nsfw.pth", model_path
            )
            print("Downloaded model H14 NSFW model to:", model_path)

        return torch.load(model_path, map_location="cpu")
