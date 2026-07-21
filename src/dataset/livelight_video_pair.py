import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class RelightVideoPairDataset(Dataset):
    """Stage 3 dataset for paired 40+40 relight sequences.

    Expected layout per sample is one image directory that contains two aligned
    clips back-to-back:
    - reference/original frames: `[reference_frame_start, reference_frame_start + N)`
    - relit target frames: `[target_frame_start, target_frame_start + N)`
    """

    def __init__(
        self,
        image_root,
        mpli_root,
        sample_list_path,
        img_size=(512, 512),
        n_sample_frames=40,
        reference_frame_start=0,
        target_frame_start=40,
        random_clip_start=False,
    ):
        super().__init__()
        self.image_root = Path(image_root)
        self.mpli_root = Path(mpli_root)
        self.samples = self._read_sample_list(sample_list_path)
        self.img_size = tuple(img_size)
        self.n_sample_frames = int(n_sample_frames)
        self.reference_frame_start = int(reference_frame_start)
        self.target_frame_start = int(target_frame_start)
        self.random_clip_start = bool(random_clip_start)

        self.clip_image_processor = CLIPImageProcessor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _read_sample_list(self, sample_list_path):
        samples = []
        for raw_line in Path(sample_list_path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            samples.append(self._parse_sample_line(line))
        return samples

    def _parse_sample_line(self, line):
        parts = line.split("\t")
        if len(parts) == 3:
            image_root, mpli_root, sample_rel = parts
            return {
                "image_root": Path(image_root),
                "mpli_root": Path(mpli_root),
                "sample_rel": Path(sample_rel),
                "display_rel": f"{Path(image_root).name}:{sample_rel}",
            }
        if len(parts) == 1:
            sample_rel = parts[0]
            return {
                "image_root": self.image_root,
                "mpli_root": self.mpli_root,
                "sample_rel": Path(sample_rel),
                "display_rel": sample_rel,
            }
        raise ValueError(
            "sample list lines must be either '<sample_rel>' or "
            "'<image_root>\\t<mpli_root>\\t<sample_rel>'"
        )

    def _load_rgb(self, path):
        return Image.open(path).convert("RGB")

    def _pack_light_sequence(self, frame_mpli_sequence):
        packed_frames = []
        for frame_mpli in frame_mpli_sequence:
            light = torch.from_numpy(frame_mpli.astype(np.float32))
            light = light.permute(0, 3, 1, 2).reshape(-1, frame_mpli.shape[1], frame_mpli.shape[2])
            if light.shape[-2:] != self.img_size:
                light = F.interpolate(
                    light.unsqueeze(0),
                    size=self.img_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            packed_frames.append(light)
        return torch.stack(packed_frames, dim=0).transpose(0, 1).contiguous()

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_rel = sample["sample_rel"]
        image_root = sample["image_root"]
        mpli_root = sample["mpli_root"]
        try:
            mpli_path = mpli_root / sample_rel / "mpli.npz"
            with np.load(mpli_path) as archive:
                mpli = archive["mpli"]
                num_target_frames = int(mpli.shape[0])
                if num_target_frames < self.n_sample_frames:
                    raise ValueError(
                        f"Sample {sample_rel} only has {num_target_frames} target frames, "
                        f"but needs {self.n_sample_frames}"
                    )
                if self.random_clip_start and num_target_frames > self.n_sample_frames:
                    clip_start = random.randint(0, num_target_frames - self.n_sample_frames)
                else:
                    clip_start = 0
                clip_end = clip_start + self.n_sample_frames
                frame_mpli_sequence = mpli[clip_start:clip_end]

            reference_pils = []
            target_pils = []
            reference_frame_ids = []
            target_frame_ids = []
            for local_idx in range(clip_start, clip_end):
                reference_frame_id = self.reference_frame_start + local_idx
                target_frame_id = self.target_frame_start + local_idx
                reference_path = image_root / sample_rel / f"{reference_frame_id:04d}.jpeg"
                target_path = image_root / sample_rel / f"{target_frame_id:04d}.jpeg"
                reference_pils.append(self._load_rgb(reference_path))
                target_pils.append(self._load_rgb(target_path))
                reference_frame_ids.append(reference_frame_id)
                target_frame_ids.append(target_frame_id)

            reference_video = torch.stack([self.image_transform(pil) for pil in reference_pils], dim=0)
            target_video = torch.stack([self.image_transform(pil) for pil in target_pils], dim=0)
            light_seq = self._pack_light_sequence(frame_mpli_sequence)
            reference_clip_video = self.clip_image_processor(images=reference_pils, return_tensors="pt").pixel_values

            return {
                "sample_rel": sample["display_rel"],
                "reference_video": reference_video,
                "target_video": target_video,
                "light_seq": light_seq,
                "reference_clip_video": reference_clip_video,
                "reference_frame_ids": torch.tensor(reference_frame_ids, dtype=torch.long),
                "target_frame_ids": torch.tensor(target_frame_ids, dtype=torch.long),
            }
        except Exception:
            next_index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(next_index)

    def __len__(self):
        return len(self.samples)
