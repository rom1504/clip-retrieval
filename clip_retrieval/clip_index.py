from autofaiss.external.quantize import Quantizer
import fire
import shutil
from glob import glob
import os
from distutils.dir_util import copy_tree
from multiprocessing import Process
import traceback
import logging


def quantize(emb_folder, index_folder, subfolder_name, index_name, max_index_memory_usage, current_memory_available):
    try:
        tmp_output_folder = index_folder+"/"+subfolder_name
        if os.path.exists(emb_folder):
            quantizer = Quantizer()
            quantizer.quantize(embeddings_path=emb_folder, output_path=tmp_output_folder, max_index_memory_usage=max_index_memory_usage, current_memory_available=current_memory_available)
            index_file = list(glob(tmp_output_folder+"/*.index"))[0]
            shutil.move(index_file, index_folder+"/"+index_name)
            os.rmdir(tmp_output_folder)
    except Exception as e:
        logging.error(traceback.format_exc())

def clip_index(embeddings_folder, index_folder, max_index_memory_usage="4G", current_memory_available="16G", copy_metadata=True, image_subfolder="img_emb", text_subfolder="text_emb"):
    p = Process(target=quantize, args=(embeddings_folder+"/"+image_subfolder, index_folder, "img_index", 
        "image.index", max_index_memory_usage, current_memory_available))
    p.start()
    p.join()
    p = Process(target=quantize, args=(embeddings_folder+"/"+text_subfolder, index_folder, "txt_index",
    "text.index", max_index_memory_usage, current_memory_available))
    p.start()
    p.join()
    if copy_metadata:
        copy_tree(embeddings_folder+"/metadata", index_folder+"/metadata")


if __name__ == '__main__':
    fire.Fire(clip_index)
