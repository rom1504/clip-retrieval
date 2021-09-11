from setuptools import setup, find_packages
from pathlib import Path

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name = 'clip_retrieval',
        packages = find_packages(),
        include_package_data = True,
        version = '2.8.1',
        license='MIT',
        description = 'Easily computing clip embeddings and building a clip retrieval system with them',
        long_description=long_description,
        long_description_content_type="text/markdown",
        entry_points={"console_scripts": ["clip-retrieval = clip_retrieval.cli:main"]},
        author = 'Romain Beaumont',
        author_email = 'romain.rom1@gmail.com',
        url = 'https://github.com/rom1504/clip-retrieval',
        data_files=[(".", ["README.md"])],
        keywords = [
            'machine learning',
            'computer vision',
            'download',
            'image',
            'dataset'
        ],
        install_requires=[
            'clip-anytorch',
            'tqdm',
            'fire',
            'torch',
            'torchvision',
            'numpy',
            'faiss-cpu',
            'flask',
            'flask_restful',
            'flask_cors',
            'pandas',
            'pyarrow',
            'autofaiss',
            'pyyaml',
            'webdataset',
            'h5py'
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
        ],
    )