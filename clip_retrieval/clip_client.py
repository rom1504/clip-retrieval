"""Clip client is a simple python module that allows you to query the backend remotely."""

import base64
import enum
import json
from pathlib import Path
from typing import Dict, List

import requests


class Modality(enum.Enum):
    IMAGE = "image"
    TEXT = "text"


class ClipClient:
    """Remotely query the CLIP backend via REST"""

    def __init__(
        self,
        url: str,
        result_dir: str = "results",
        indice_name: str = "laion5B",
        use_mclip: bool = False,
        aesthetic_score: int = 9,
        aesthetic_weight: float = 0.5,
        modality: Modality = Modality.IMAGE,
        num_images: int = 40,
        num_result_ids: int = 3000,
    ):
        """
        modality: which "modality" to search over, text or image, defaults to image.
        num_images: number of images to return (may be less).
        indice_name: which indice to search over, laion5B or laion_400m.
        num_result_ids: number of result ids to be returned.
        use_mclip: whether to use mclip, a multilingual version of clip.
        aesthetic_score: ranking score for aesthetic, higher is prettier.
        aesthetic_weight: weight of the aesthetic score, between 0 and 1.
        """
        self.url = url
        self.result_dir = Path(result_dir)
        assert not self.result_dir.exists(), f"{self.result_dir} already exists. Move or delete it and try again."
        self.result_dir.mkdir(parents=True)
        self.indice_name = indice_name
        self.use_mclip = use_mclip
        self.aesthetic_score = aesthetic_score
        self.aesthetic_weight = aesthetic_weight
        self.modality = modality.value
        self.num_images = num_images
        self.num_result_ids = num_result_ids

    def query(
        self,
        text: str = None,
        image: str = None,
    ) -> List[Dict]:
        """
        Given text or image/s, search for other captions/images that are semantically similar.

        Args:
            text: text to be searched semantically.
            image: base64 string of image to be searched semantically

        Returns:
            List of dictionaries containing the results.
        """
        if text and image:
            raise ValueError("Only one of text or image can be provided.")
        if text:
            print(f"Searching for {text}")
            return self.__search_knn_api__(text=text)
        elif image:
            if image.startswith("http"):
                print(f"Searching via image at url {image}")
                return self.__search_knn_api__(image_url=image)
            else:
                print(f"Searching for image at path {image}")
                assert Path(image).exists(), f"{image} does not exist."
                return self.__search_knn_api__(image=image)
        else:
            raise ValueError("Either text or image must be provided.")

    def __search_knn_api__(
        self,
        text: str = None,
        image: str = None,
        image_url: str = None,
    ) -> List:
        """
        This function is used to send the request to the knn service.
        It represents a direct API call and should not be called directly outside the package.

        Args:
            text: text to be searched semantically.
            image: base64 string of image to be searched semantically.

        Returns:
            List of dictionaries containing the results.
        """
        if image:
            # Convert image to base64 string
            with open(image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                image = str(encoded_string.decode("utf-8"))
        return requests.post(
            self.url,
            data=json.dumps(
                {
                    "text": text,
                    "image": image,
                    "image_url": image_url,
                    "deduplicate": True,
                    "use_safety_model": True,
                    "use_violence_detector": True,
                    "indice_name": self.indice_name,
                    "use_mclip": self.use_mclip,
                    "aesthetic_score": self.aesthetic_score,
                    "aesthetic_weight": self.aesthetic_weight,
                    "modality": self.modality,
                    "num_images": self.num_images,
                    "num_result_ids": self.num_result_ids,
                }
            ),
        ).json()[
            0 : self.num_images
        ]  # The "last half" is used for formatting on the website.
