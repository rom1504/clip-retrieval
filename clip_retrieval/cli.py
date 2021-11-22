"""cli entry point"""

from clip_retrieval.clip_back import clip_back
from clip_retrieval.clip_inference import clip_inference
from clip_retrieval.clip_filter import clip_filter
from clip_retrieval.clip_index import clip_index
from clip_retrieval.clip_end2end import clip_end2end
from clip_retrieval.clip_front import clip_front
import fire


def main():
    """Main entry point"""
    fire.Fire(
        {
            "back": clip_back,
            "inference": clip_inference,
            "index": clip_index,
            "filter": clip_filter,
            "end2end": clip_end2end,
            "front": clip_front,
        }
    )
