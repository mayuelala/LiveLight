import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from src.utils.geometry import depth_to_normals_from_camera_depth, scale_intrinsics


class RelightImageDataset(Dataset):
    def __init__(
        self,
        image_root,
        mpli_root,
        sample_list_path=None,
        sample_list_paths=None,
        img_size=(512, 512),
        source_frame_id=0,
        lit_frame_start=1,
        enable_geometry=False,
    ):
        super().__init__()
        self.image_root = Path(image_root)
        self.mpli_root = Path(mpli_root)
        self.samples = self._read_sample_lists(sample_list_path=sample_list_path, sample_list_paths=sample_list_paths)
        self.img_size = tuple(img_size)
        self.source_frame_id = int(source_frame_id)
        self.lit_frame_start = int(lit_frame_start)
        self.enable_geometry = bool(enable_geometry)

        self.known_data_roots = []

        self.clip_image_processor = CLIPImageProcessor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _normalize_root_path(self, path):
        return Path(path)

    def _normalize_sample_list_paths(self, sample_list_path=None, sample_list_paths=None):
        paths = []
        if sample_list_path is not None:
            paths.append(sample_list_path)
        if sample_list_paths is not None:
            paths.extend(sample_list_paths)
        if not paths:
            raise ValueError("At least one sample list path must be provided")
        return [Path(path) for path in paths]

    def _read_sample_lists(self, sample_list_path=None, sample_list_paths=None):
        samples = []
        for path in self._normalize_sample_list_paths(sample_list_path, sample_list_paths):
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                samples.append(self._parse_sample_line(line))
        return samples

    def _parse_sample_line(self, line):
        parts = line.split("\t")
        if len(parts) == 4:
            image_root, mpli_root, sample_rel, pair_offset_str = parts
            return {
                "image_root": self._normalize_root_path(image_root),
                "mpli_root": self._normalize_root_path(mpli_root),
                "sample_rel": Path(sample_rel),
                "display_rel": f"{Path(image_root).name}:{sample_rel}",
                "pair_offset": int(pair_offset_str),
            }
        if len(parts) == 3:
            image_root, mpli_root, sample_rel = parts
            return {
                "image_root": self._normalize_root_path(image_root),
                "mpli_root": self._normalize_root_path(mpli_root),
                "sample_rel": Path(sample_rel),
                "display_rel": f"{Path(image_root).name}:{sample_rel}",
            }
        if len(parts) == 1:
            entry = self._normalize_root_path(parts[0])
            if entry.is_absolute() and entry.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                image_root = entry.parents[3]
                sample_rel = entry.parent.relative_to(image_root)
                return {
                    "image_root": image_root,
                    "mpli_root": None,
                    "sample_rel": sample_rel,
                    "display_rel": f"{image_root.name}:{sample_rel}",
                    "source_frame_id_override": int(entry.stem),
                }
            sample_rel = Path(parts[0])
            return {
                "image_root": self.image_root,
                "mpli_root": self.mpli_root,
                "sample_rel": sample_rel,
                "display_rel": sample_rel.as_posix(),
            }
        raise ValueError(
            "sample list lines must be either '<sample_rel>', '<absolute_source_frame_path>', or "
            "'<image_root>\\t<mpli_root>\\t<sample_rel>'"
        )

    def _load_rgb(self, path):
        return Image.open(path).convert("RGB")

    def _pack_light(self, frame_mpli):
        light = torch.from_numpy(frame_mpli.astype(np.float32))
        light = light.permute(0, 3, 1, 2).reshape(-1, frame_mpli.shape[1], frame_mpli.shape[2])
        if light.shape[-2:] != self.img_size:
            light = F.interpolate(
                light.unsqueeze(0),
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return light

    def _try_roots(self, candidates):
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = self._normalize_root_path(candidate)
            if candidate.exists():
                return candidate
        return None

    def _candidate_data_roots(self, image_root):
        candidates = [image_root.parent]
        for root in self.known_data_roots:
            candidates.append(root)
        unique = []
        seen = set()
        for root in candidates:
            key = str(root)
            if key not in seen:
                seen.add(key)
                unique.append(root)
        return unique

    def _resolve_named_root(self, image_root, prefix, aggregate_name=None):
        basename = image_root.name
        candidates = []

        if basename == "image":
            sibling_name = {
                "Condition_": "condition",
                "Labels_": "label",
                "DepthPPD_": "depth",
                "NormalPPD_": "normal",
            }[prefix]
            candidates.append(image_root.parent / sibling_name)
        elif basename.startswith("MovieRenders_"):
            swapped = basename.replace("MovieRenders_", prefix, 1)
            for data_root in self._candidate_data_roots(image_root):
                candidates.append(data_root / swapped)

            if aggregate_name is not None:
                for data_root in self._candidate_data_roots(image_root):
                    candidates.append(data_root / aggregate_name)

        return self._try_roots(candidates)

    def _resolve_mpli_root(self, sample):
        explicit_root = sample.get("mpli_root")
        if explicit_root is not None and explicit_root.exists():
            return explicit_root
        return self._resolve_named_root(sample["image_root"], "Condition_")

    def _resolve_label_root(self, sample):
        return self._resolve_named_root(sample["image_root"], "Labels_")

    def _resolve_depth_root(self, sample):
        image_root = sample["image_root"]
        basename = image_root.name
        aggregate_name = None
        if "_F80_Pair_Character_NearMid_" in basename:
            aggregate_name = "DepthPPD_Relight_F80_Pair_Character_NearMid_DarkstrictOev1_Source40"
        return self._resolve_named_root(image_root, "DepthPPD_", aggregate_name=aggregate_name)

    def _resolve_normal_root(self, sample):
        image_root = sample["image_root"]
        basename = image_root.name
        aggregate_name = None
        if "_F80_Pair_Character_NearMid_" in basename:
            aggregate_name = "NormalPPD_Relight_F80_Pair_Character_NearMid_DarkstrictOev1_Source40"
        return self._resolve_named_root(image_root, "NormalPPD_", aggregate_name=aggregate_name)

    def _label_filename(self, sample_rel):
        seq_name = "_".join(sample_rel.parts[-3:])
        return f"Seq_{seq_name}.json"

    def _load_precomputed_normal(self, sample, source_frame_id):
        normal_root = self._resolve_normal_root(sample)
        if normal_root is None:
            return None, None

        sample_rel = sample["sample_rel"]
        normal_candidates = [
            normal_root / sample_rel,
            normal_root / sample["image_root"].name / sample_rel,
        ]
        normal_dir = self._try_roots(normal_candidates)
        if normal_dir is None:
            return None, None

        normal_path = normal_dir / f"{source_frame_id:04d}.npy"
        if not normal_path.exists():
            return None, None

        normal_np = np.load(normal_path)
        if normal_np.ndim == 3 and normal_np.shape[-1] == 3:
            normal = torch.from_numpy(normal_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        elif normal_np.ndim == 3 and normal_np.shape[0] == 3:
            normal = torch.from_numpy(normal_np.astype(np.float32)).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected normal shape {normal_np.shape} for {normal_path}")

        normal_valid_path = normal_dir / f"{source_frame_id:04d}_valid.npy"
        if normal_valid_path.exists():
            normal_valid_np = np.load(normal_valid_path)
            if normal_valid_np.ndim == 2:
                normal_valid = torch.from_numpy(normal_valid_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            elif normal_valid_np.ndim == 3 and normal_valid_np.shape[0] == 1:
                normal_valid = torch.from_numpy(normal_valid_np.astype(np.float32)).unsqueeze(0)
            else:
                raise ValueError(
                    f"Unexpected normal valid shape {normal_valid_np.shape} for {normal_valid_path}"
                )
        else:
            normal_valid = (torch.linalg.vector_norm(normal, dim=1, keepdim=True) > 1e-8).float()

        return normal, normal_valid

    def _load_geometry(self, sample, source_frame_id):
        label_root = self._resolve_label_root(sample)
        depth_root = self._resolve_depth_root(sample)
        if label_root is None or depth_root is None:
            raise FileNotFoundError(f"Could not resolve label/depth roots for {sample['display_rel']}")

        sample_rel = sample["sample_rel"]
        label_path = label_root / sample_rel / self._label_filename(sample_rel)
        depth_candidates = [
            depth_root / sample_rel,
            depth_root / sample["image_root"].name / sample_rel,
        ]
        depth_dir = self._try_roots(depth_candidates)
        if depth_dir is None:
            raise FileNotFoundError(f"Could not resolve depth directory for {sample['display_rel']}")

        depth_path = depth_dir / f"{source_frame_id:04d}.npy"
        depth_valid_path = depth_dir / f"{source_frame_id:04d}_valid.npy"

        with label_path.open("r", encoding="utf-8") as handle:
            label = json.load(handle)
        camera = label["camera"]
        intrinsics = torch.tensor([camera["fx"], camera["fy"], camera["cx"], camera["cy"]], dtype=torch.float32)
        src_hw = (int(camera["image_height"]), int(camera["image_width"]))

        depth = torch.from_numpy(np.load(depth_path).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if depth_valid_path.exists():
            valid = torch.from_numpy(np.load(depth_valid_path).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        else:
            valid = (depth > 1e-6).float()

        if depth.shape[-2:] != self.img_size:
            depth = F.interpolate(depth, size=self.img_size, mode="nearest")
            valid = F.interpolate(valid, size=self.img_size, mode="nearest")

        depth = depth[0]
        valid = (valid[0] > 0.5).float()
        scaled_intrinsics = scale_intrinsics(intrinsics, src_hw=src_hw, dst_hw=self.img_size)[0]

        normal, normal_valid = self._load_precomputed_normal(sample, source_frame_id=source_frame_id)
        if normal is not None:
            if normal.shape[-2:] != self.img_size:
                normal = F.interpolate(normal, size=self.img_size, mode="nearest")
                normal_valid = F.interpolate(normal_valid, size=self.img_size, mode="nearest")
            normal = normal[0]
            normal_valid = normal_valid[0] > 0.5
            normal = torch.where(normal_valid.expand_as(normal), normal, torch.zeros_like(normal))
        else:
            normal, normal_valid = depth_to_normals_from_camera_depth(
                depth.unsqueeze(0),
                scaled_intrinsics.unsqueeze(0),
                valid_mask=valid.unsqueeze(0) > 0.5,
            )
            normal = normal[0]

        geometry_mask = (valid.unsqueeze(0) > 0.5) & normal_valid
        return {
            "gt_depth": depth,
            "gt_normal": normal,
            "gt_depth_valid_mask": valid,
            "gt_geometry_valid_mask": geometry_mask[0].float(),
            "geometry_intrinsics": scaled_intrinsics,
        }

    def __getitem__(self, index):
        last_error = None
        curr_index = index
        for _ in range(16):
            sample = self.samples[curr_index]
            sample_rel = sample["sample_rel"]
            image_root = self._normalize_root_path(sample["image_root"])
            source_frame_id = int(sample.get("source_frame_id_override", self.source_frame_id))

            try:
                source_path = image_root / sample_rel / f"{source_frame_id:04d}.jpeg"
                source_pil = self._load_rgb(source_path)

                mpli_root = self._resolve_mpli_root(sample)
                if mpli_root is None:
                    raise FileNotFoundError(f"Could not resolve MPLI root for {sample['display_rel']}")

                mpli_path = mpli_root / sample_rel / "mpli.npz"
                with np.load(mpli_path) as archive:
                    mpli = archive["mpli"]
                    num_lit_frames = int(mpli.shape[0])
                    lit_frame_index = random.randint(0, num_lit_frames - 1)
                    frame_mpli = mpli[lit_frame_index]

                pair_offset = sample.get("pair_offset")
                if pair_offset is not None:
                    source_frame_id = lit_frame_index
                    source_path = image_root / sample_rel / f"{source_frame_id:04d}.jpeg"
                    source_pil = self._load_rgb(source_path)
                    target_frame_id = lit_frame_index + pair_offset
                else:
                    target_frame_id = self.lit_frame_start + lit_frame_index
                target_path = image_root / sample_rel / f"{target_frame_id:04d}.jpeg"
                target_pil = self._load_rgb(target_path)

                source_img = self.image_transform(source_pil)
                target_img = self.image_transform(target_pil)
                light = self._pack_light(frame_mpli)
                clip_image = self.clip_image_processor(images=source_pil, return_tensors="pt").pixel_values[0]

                result = {
                    "sample_rel": sample["display_rel"],
                    "source_img": source_img,
                    "target_img": target_img,
                    "light": light,
                    "clip_image": clip_image,
                    "target_frame_id": target_frame_id,
                }
                if self.enable_geometry:
                    result.update(self._load_geometry(sample, source_frame_id=source_frame_id))
                return result
            except Exception as exc:
                last_error = exc
                curr_index = random.randint(0, len(self.samples) - 1)

        raise RuntimeError(f"Failed to load a valid relight sample after repeated retries: {last_error}")

    def __len__(self):
        return len(self.samples)
