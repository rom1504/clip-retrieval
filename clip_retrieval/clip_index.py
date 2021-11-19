from autofaiss import build_index
import fire
import os
from distutils.dir_util import copy_tree
from multiprocessing import Process

def quantize(emb_folder, index_folder, index_name, max_index_memory_usage, current_memory_available, nb_cores):
    try:
        if os.path.exists(emb_folder):
            build_index(
                embeddings_path=emb_folder,
                index_path=index_folder+"/"+index_name+".index",
                index_infos_path=index_folder+"/"+index_name+".json", 
                max_index_memory_usage=max_index_memory_usage,
                current_memory_available=current_memory_available,
                nb_cores=nb_cores)
    except Exception as e:
        print(e)

def clip_index(embeddings_folder, index_folder, max_index_memory_usage="4G", current_memory_available="16G", 
            copy_metadata=True, image_subfolder="img_emb", text_subfolder="text_emb", nb_cores=None):
    p = Process(target=quantize, args=(embeddings_folder+"/"+image_subfolder, index_folder, 
        "image", max_index_memory_usage, current_memory_available, nb_cores))
    p.start()
    p.join()
    p = Process(target=quantize, args=(embeddings_folder+"/"+text_subfolder, index_folder,
    "text", max_index_memory_usage, current_memory_available, nb_cores))
    p.start()
    p.join()
    if copy_metadata:
        copy_tree(embeddings_folder+"/metadata", index_folder+"/metadata")


if __name__ == '__main__':
    fire.Fire(clip_index)
