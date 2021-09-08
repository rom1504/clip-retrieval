from autofaiss.external.quantize import Quantizer
import fire
import shutil
from glob import glob
import os
from distutils.dir_util import copy_tree

def clip_index(embeddings_folder, index_folder, max_index_memory_usage="4G", current_memory_available="16G"):
    quantizer = Quantizer()
    if os.path.exists(embeddings_folder+"/img_emb"):
        quantizer.quantize(embeddings_path=embeddings_folder+"/img_emb", output_path=index_folder+"/img_index", max_index_memory_usage=max_index_memory_usage, current_memory_available=current_memory_available)
        index_file = list(glob(index_folder+"/img_index/*.index"))[0]
        shutil.move(index_file, index_folder+"/image.index")
        os.rmdir(index_folder+"/img_index")
    if os.path.exists(embeddings_folder+"/text_emb"):
        quantizer.quantize(embeddings_path=embeddings_folder+"/text_emb", output_path=index_folder+"/txt_index", max_index_memory_usage=max_index_memory_usage, current_memory_available=current_memory_available)
        index_file = list(glob(index_folder+"/txt_index/*.index"))[0]
        shutil.move(index_file, index_folder+"/text.index")
        os.rmdir(index_folder+"/txt_index")
    copy_tree(embeddings_folder+"/metadata", index_folder+"/metadata")


if __name__ == '__main__':
    fire.Fire(clip_index)
