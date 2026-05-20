import json
import logging
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = List[Dict[str, Any]]  # List of annotations for the image


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def length(self) -> int:
        # These are approximate values, will be updated by actual dataset length
        split_lengths = {
            _Split.TRAIN: 118_000,
            _Split.VAL: 5_000,
        }
        return split_lengths[self]

    def get_dirname(self) -> str:
        return f"{self.value}2017"

    def get_annotation_filename(self) -> str:
        return f"annotations/instances_{self.value}2017.json"
    
    def get_caption_filename(self) -> str:
        return f"annotations/captions_{self.value}2017.json"


class MSCOCO(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "MSCOCO.Split",
        root: str,
        extra: str,
        use_captions: bool = True,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        self._use_captions = use_captions

        self._entries = None
        self._captions = None
        self._instances = None

    @property
    def split(self) -> "MSCOCO.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(os.path.dirname(extra_full_path), exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _captions_path(self) -> str:
        return f"captions-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            try:
                self._entries = self._load_extra(self._entries_path)
            except (FileNotFoundError, IOError):
                logger.info(f"Entries not found at {self._entries_path}, creating them...")
                self._dump_entries()
                self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_captions(self) -> Optional[np.ndarray]:
        if not self._use_captions:
            return None
            
        if self._captions is None:
            try:
                self._captions = self._load_extra(self._captions_path)
            except (FileNotFoundError, IOError):
                logger.info(f"Captions not found at {self._captions_path}, creating them...")
                self._dump_captions()
                self._captions = self._load_extra(self._captions_path)
        return self._captions

    def _load_coco_annotations(self, annotation_file: str) -> Dict:
        """Load COCO annotation file."""
        annotation_path = os.path.join(self.root, annotation_file)
        try:
            with open(annotation_path, 'r') as f:
                return json.load(f)
        except OSError as e:
            raise RuntimeError(f'Cannot read annotation file "{annotation_path}"') from e

    def get_image_data(self, index: int) -> bytes:
        """Get the raw image data at the given index."""
        entries = self._get_entries()
        image_id = entries[index]["image_id"]
        file_name = entries[index]["file_name"]
        
        image_dir = self.split.get_dirname()
        image_path = os.path.join(self.root, image_dir, file_name)
        
        with open(image_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Target:
        """Get the target (captions or annotations) for the image at the given index."""
        entries = self._get_entries()
        image_id = entries[index]["image_id"]
        
        if self._use_captions:
            captions = self._get_captions()
            if captions is not None:
                # Find all captions for this image_id
                return [cap for cap in captions if cap["image_id"] == image_id]
        
        # If not using captions or captions not found, return empty list
        return []

    def get_caption(self, index: int) -> Optional[str]:
        """Get a single caption for the image at the given index."""
        if not self._use_captions:
            return None
            
        captions = self.get_target(index)
        if captions and "caption" in captions[0]:
            return captions[0]["caption"]
        return None

    def get_image_id(self, index: int) -> int:
        """Get the COCO image ID for the image at the given index."""
        entries = self._get_entries()
        return entries[index]["image_id"]

    def __len__(self) -> int:
        entries = self._get_entries()
        return len(entries)

    def _dump_entries(self) -> None:
        """Create and save the entries array from COCO annotations."""
        split = self.split
        
        # Load instance annotations to get image info
        instances_file = split.get_annotation_filename()
        logger.info(f'Loading COCO instances from "{instances_file}"')
        instances = self._load_coco_annotations(instances_file)
        
        # Extract image information
        images = instances["images"]
        sample_count = len(images)
        
        # Create entries array
        dtype = np.dtype([
            ("image_id", "<i4"),
            ("file_name", "U100"),  # Assuming file names are less than 100 chars
            ("width", "<i4"),
            ("height", "<i4"),
        ])
        entries_array = np.empty(sample_count, dtype=dtype)
        
        for i, image in enumerate(images):
            entries_array[i] = (
                image["id"],
                image["file_name"],
                image["width"],
                image["height"],
            )
        
        logger.info(f'Saving {sample_count} entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_captions(self) -> None:
        """Create and save the captions array from COCO annotations."""
        if not self._use_captions:
            return
            
        split = self.split
        
        # Load caption annotations
        captions_file = split.get_caption_filename()
        logger.info(f'Loading COCO captions from "{captions_file}"')
        captions_data = self._load_coco_annotations(captions_file)
        
        # Extract captions
        captions = captions_data["annotations"]
        
        # Find the maximum caption length
        max_caption_length = max(len(cap["caption"]) for cap in captions)
        
        # Create captions array
        dtype = np.dtype([
            ("id", "<i4"),
            ("image_id", "<i4"),
            ("caption", f"U{max_caption_length}"),
        ])
        captions_array = np.empty(len(captions), dtype=dtype)
        
        for i, caption in enumerate(captions):
            captions_array[i] = (
                caption["id"],
                caption["image_id"],
                caption["caption"],
            )
        
        logger.info(f'Saving {len(captions)} captions to "{self._captions_path}"')
        self._save_extra(captions_array, self._captions_path)

    def dump_extra(self) -> None:
        """Dump all extra data needed for the dataset."""
        self._dump_entries()
        if self._use_captions:
            self._dump_captions()