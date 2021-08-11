from clip_retrieval.clip_back import clip_back
from clip_retrieval.clip_batch import clip_batch
from clip_retrieval.clip_filter import clip_filter
import fire
import logging


def main():
    """Main entry point"""
    fire.Fire(
        {
            "back": clip_back,
            "batch": clip_batch,
            "filter": clip_filter
        }
    )