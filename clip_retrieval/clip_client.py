"""Clip client is a simple python module that allows you to query the backend remotely."""

import base64
import enum
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests


class Modality(enum.Enum):
    IMAGE = "image"
    TEXT = "text"


class ClipClient:
    """Remotely query the CLIP backend via REST"""

    def __init__(
        self,
        url: str,
        indice_name: str,
        use_mclip: bool = False,
        aesthetic_score: int = 9,
        aesthetic_weight: float = 0.5,
        modality: Modality = Modality.IMAGE,
        num_images: int = 40,
        deduplicate: bool = True,
        use_safety_model: bool = True,
        use_violence_detector: bool = True,
    ):
        """
        url: (required) URL of the backend.
        indice_name: (required) which indice to search over e.g. "laion5B" or "laion_400m".
        use_mclip: (optional) whether to use mclip, a multilingual version of clip. Default is False.
        aesthetic_score: (optional) ranking score for aesthetic, higher is prettier. Default is 9.
        aesthetic_weight: (optional) weight of the aesthetic score, between 0 and 1. Default is 0.5.
        modality: (optional) Search modality. One of Modality.IMAGE or Modality.TEXT. Default is Modality.IMAGE.
        num_images: (optional) Number of images to return. Default is 40.
        deduplicate: (optional) Whether to deduplicate the result by image embedding. Default is true.
        use_safety_model: (optional) Whether to remove unsafe images. Default is true.
        use_violence_detector: (optional) Whether to remove images with violence. Default is true.
        """
        self.url = url
        self.indice_name = indice_name
        self.use_mclip = use_mclip
        self.aesthetic_score = aesthetic_score
        self.aesthetic_weight = aesthetic_weight
        self.modality = modality.value
        self.num_images = num_images
        self.deduplicate = deduplicate
        self.use_safety_model = use_safety_model
        self.use_violence_detector = use_violence_detector

    def query(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        embedding_input: Optional[list] = None,
    ) -> List[Dict]:
        """
        Given text or image/s, search for other captions/images that are semantically similar.

        Args:
            text: text to be searched semantically.
            image: base64 string of image to be searched semantically

        Returns:
            List of dictionaries containing the results in the form of:
            [
                {
                    "id": 42,
                    "similarity": 0.323424523424,
                    "url": "https://example.com/image.jpg",
                    "caption": "This is a caption",
                },
                ...
            ]
        """
        if text and image:
            raise ValueError("Only one of text or image can be provided.")
        if text:
            return self.__search_knn_api__(text=text)
        elif image:
            if image.startswith("http"):
                return self.__search_knn_api__(image_url=image)
            else:
                assert Path(image).exists(), f"{image} does not exist."
                return self.__search_knn_api__(image=image)
        elif embedding_input:
            return self.__search_knn_api__(embedding_input=embedding_input)
        else:
            raise ValueError("Either text or image must be provided.")

    def __search_knn_api__(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        image_url: Optional[str] = None,
        embedding_input: Optional[list] = None,
    ) -> List:
        """
        This function is used to send the request to the knn service.
        It represents a direct API call and should not be called directly outside the package.

        Args:
            text: text to be searched semantically.
            image: base64 string of image to be searched semantically.
            image_url: url of the image to be searched semantically.
            embedding_input: embedding input

        Returns:
            List of dictionaries containing the results in the form of:
            [
                {
                    "id": 42,
                    "similarity": 0.323424523424,
                    "url": "https://example.com/image.jpg",
                    "caption": "This is a caption",
                },
                ...
            ]

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
                    "embedding_input": embedding_input,
                    "deduplicate": self.deduplicate,
                    "use_safety_model": self.use_safety_model,
                    "use_violence_detector": self.use_violence_detector,
                    "indice_name": self.indice_name,
                    "use_mclip": self.use_mclip,
                    "aesthetic_score": self.aesthetic_score,
                    "aesthetic_weight": self.aesthetic_weight,
                    "modality": self.modality,
                    "num_images": self.num_images,
                    # num_results_ids is hardcoded to the num_images parameter.
                    "num_result_ids": self.num_images,
                }
            ),
            timeout=3600,
        ).json()
