#!pip install clip-anytorch faiss-cpu fire
import torch
import clip
from PIL import Image
from glob import glob
import fire
  
from pathlib import Path

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import faiss
import os

class ImageDataset(Dataset):
    def __init__(self,
                 preprocess, 
                 folder,
                 description_index=0,
                 enable_text=True,
                 enable_image=True
                 ):
        super().__init__()
        path = Path(folder)
        self.enable_text = enable_text
        self.enable_image = enable_image

        if self.enable_text:
            text_files = [*path.glob('**/*.txt')]
            text_files = {text_file.stem: text_file for text_file in text_files}
        if self.enable_image:
            image_files = [
                *path.glob('**/*.png'), *path.glob('**/*.jpg'),
                *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
            ]
            image_files = {image_file.stem: image_file for image_file in image_files}

        if enable_text and enable_image:
            keys = (image_files.keys() & text_files.keys())
        elif enable_text:
            keys = text_files.keys()
        elif enable_image:
            keys = image_files.keys()

        self.keys = list(keys)
        if self.enable_text:
            self.tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]
            self.text_files = {k: v for k, v in text_files.items() if k in keys}
            self.description_index = description_index
        if self.enable_image:
            self.image_files = {k: v for k, v in image_files.items() if k in keys}
            self.image_transform = preprocess

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]

        if self.enable_image:
            image_file = self.image_files[key]
            image_tensor = self.image_transform(PIL.Image.open(image_file))


        if self.enable_text:
            text_file = self.text_files[key]
            descriptions = text_file.read_text().split('\n')
            description = descriptions[self.description_index]
            tokenized_text  = self.tokenizer(description)

        return {"image_tensor": image_tensor, "text_tokens": tokenized_text, "image_filename": str(image_file), "text": description}
    

def main(dataset_path, output_folder, batch_size=256, num_prepro_workers=8, description_index=0, enable_text=True, enable_image=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    data = DataLoader(ImageDataset(preprocess, dataset_path, description_index=description_index, enable_text=enable_text, enable_image=enable_image), \
        batch_size=batch_size, shuffle=False, num_workers=num_prepro_workers, pin_memory=True, prefetch_factor=2)
    if enable_image:
        image_embeddings = []
        image_names = []
    if enable_text:
        text_embeddings = []
        descriptions = []

    for i, item in enumerate(tqdm(data)):
        with torch.no_grad():
            if enable_image:
                image_features = model.encode_image(item["image_tensor"].cuda())
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embeddings.append(image_features.cpu().numpy())
                image_names.extend(item["image_filename"])
            if enable_text:
                text_features = model.encode_text(item["text_tokens"].cuda())
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_embeddings.append(text_features.cpu().numpy())
                descriptions.extend(item["text"])

    if enable_image:
        img_emb_mat = np.concatenate(image_embeddings)
        np.save(output_folder + "/img_emb", img_emb_mat)
        with open(output_folder + "/image_list", "w") as f:
            f.write("\n".join(image_names)+"\n")
        img_index = faiss.IndexFlatIP(img_emb_mat.shape[1])
        img_index.add(img_emb_mat.astype("float32"))
        faiss.write_index(img_index, output_folder +"/image.index")

    if enable_text:
        text_emb_mat = np.concatenate(text_embeddings)
        np.save(output_folder + "/text_emb", text_emb_mat)

        with open(output_folder + "/description_list", "w") as f:
            f.write("\n".join(descriptions)+"\n")

        text_index = faiss.IndexFlatIP(text_emb_mat.shape[1])
        text_index.add(text_emb_mat.astype("float32"))
        faiss.write_index(text_index, output_folder +"/text.index")
    
if __name__ == '__main__':
  fire.Fire(main)
