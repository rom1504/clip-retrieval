from setuptools import setup, find_packages
from pathlib import Path
import os

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    import os

    def package_files(directory):
        paths = []
        for (path, _, filenames) in os.walk(directory):
            for filename in filenames:
                paths.append(os.path.join("..", path, filename))
        return paths

    extra_files = package_files("front/build")

    def _read_reqs(relpath):
        fullpath = os.path.join(os.path.dirname(__file__), relpath)
        with open(fullpath) as f:
            return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

    REQUIREMENTS = _read_reqs("requirements.txt")

    setup(
        name="clip_retrieval",
        packages=find_packages(),
        package_data={"": extra_files},
        include_package_data=True,
        version="2.25.0",
        license="MIT",
        description="Easily computing clip embeddings and building a clip retrieval system with them",
        long_description=long_description,
        long_description_content_type="text/markdown",
        entry_points={"console_scripts": ["clip-retrieval = clip_retrieval.cli:main"]},
        author="Romain Beaumont",
        author_email="romain.rom1@gmail.com",
        url="https://github.com/rom1504/clip-retrieval",
        data_files=[(".", ["README.md"]),],
        keywords=["machine learning", "computer vision", "download", "image", "dataset"],
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
        ],
    )
