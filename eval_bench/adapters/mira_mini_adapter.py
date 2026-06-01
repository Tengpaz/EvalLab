"""
EvalLab adapter for EasyLab-JiT MiraMini multi-view image generation models.

Supports three model variants:
  mira_mini               — original MiraMiniTransformer (transformer_mira_mini.py)
  mira_mini_prope         — PRoPE variant (transformer_mira_mini_prope.py)
  mira_mini_prope_register — PRoPE + Register Tokens (transformer_mira_mini_prope_register.py)

Required extra_args in the model YAML config:
  easylab_path (str): absolute path to the EasyLab-JiT repository root
  model_type   (str): one of the three variant names above

Optional extra_args:
  img_size            (int,  default 256): resize images to this square resolution
  num_sampling_steps  (int,  default 50):  Euler steps for diffusion sampling
  dtype               (str,  default "float32"): torch dtype for inference ("float32" or "bfloat16")
  model_params        (dict): kwargs forwarded verbatim to the model constructor
  diffusion_params    (dict): overrides for FlowMatching defaults

Checkpoint format (EasyLab-JiT native backend):
  The checkpoint .pt file must contain a "model" key with the model state dict,
  as written by easylab.training.backends.native.checkpoint.save_checkpoint.
  Typical path: outputs/{name}/{tag}/ckpts/step_{step:07d}.pt

Camera convention:
  eval_bench cameras use opengl_c2w (transform_matrix is camera-to-world).
  The model expects world-to-camera matrices; this adapter inverts them.

View count constraint:
  len(input_ids) + len(target_ids) must equal the model's num_views parameter
  (8 for the DL3DV-trained L/16 variants).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _strip_state_dict_prefix(state_dict: dict) -> dict:
    """Strip common wrapper prefixes from checkpoint state dicts.

    EasyLab-JiT saves the full NativeSystem state dict (not just system.model),
    and DDP training adds a ``module.`` wrapper. The resulting keys look like
    ``model.module.<param>`` instead of ``<param>``. This function detects
    and removes the longest common prefix so the state dict can be loaded
    directly into the bare model.
    """
    if not state_dict:
        return state_dict
    # Try prefixes in order from longest to shortest
    for prefix in ("model.module.", "module.model.", "model.", "module."):
        if all(k.startswith(prefix) for k in state_dict):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


class MiraMiniAdapter:
    """Unified EvalLab adapter for all EasyLab-JiT MiraMini model variants."""

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def setup(self, model_config: dict[str, Any]) -> None:
        extra = dict(model_config.get("extra_args") or {})

        # ---- EasyLab import path ----
        easylab_path = extra.get("easylab_path")
        if easylab_path:
            easylab_path = str(Path(easylab_path).expanduser())
            if easylab_path not in sys.path:
                sys.path.insert(0, easylab_path)

        # ---- Inference settings ----
        self.img_size = int(extra.get("img_size", 256))
        self.num_sampling_steps = int(extra.get("num_sampling_steps", 50))
        dtype_str = extra.get("dtype", "float32")
        self.dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

        device_str = model_config.get("device", "cuda:0")
        self.device = torch.device(device_str)

        # ---- Model construction ----
        model_type = extra.get("model_type", "mira_mini_prope")
        model_params = dict(extra.get("model_params") or {})
        model_params.setdefault("input_size", self.img_size)

        if model_type == "mira_mini":
            from easylab.models.mira_mini.transformer_mira_mini import MiraMiniTransformer
            self.model = MiraMiniTransformer(**model_params)
        elif model_type == "mira_mini_prope":
            from easylab.models.mira_mini.transformer_mira_mini_prope import MiraMiniTransformer
            self.model = MiraMiniTransformer(**model_params)
        elif model_type == "mira_mini_prope_register":
            from easylab.models.mira_mini.transformer_mira_mini_prope_register import (
                MiraMiniTransformerWithRegisters,
            )
            self.model = MiraMiniTransformerWithRegisters(**model_params)
        else:
            raise ValueError(
                f"Unknown model_type: {model_type!r}. "
                "Expected one of: mira_mini, mira_mini_prope, mira_mini_prope_register"
            )

        # ---- Flow matching ----
        from easylab.diffusion.flow_matching import FlowMatching

        diffusion_defaults = dict(
            pred_mode="x0_to_v",
            t_schedule="logit_normal",
            P_mean=0.8,
            P_std=0.8,
            t_eps=0.05,
            noise_scale=2.0,
        )
        diffusion_defaults.update(dict(extra.get("diffusion_params") or {}))
        self.flow_matching = FlowMatching(**diffusion_defaults)

        # ---- Load checkpoint ----
        weights = model_config.get("weights")
        if weights:
            weights_path = Path(weights).expanduser()
            ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
            model_state = ckpt.get("model", ckpt)
            model_state = _strip_state_dict_prefix(model_state)
            missing, unexpected = self.model.load_state_dict(model_state, strict=True)
            # strict=True; any remaining mismatch is a real error

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict(self, batch: dict[str, Any]) -> dict[int, Image.Image]:
        """Generate novel target views for one scene.

        Args:
            batch: eval_bench batch dict containing input_images, input_cameras,
                   target_cameras, and target_ids.

        Returns:
            Dict mapping each target_id to a predicted PIL.Image.
        """
        input_images: list[Image.Image] = batch["input_images"]
        input_cameras: list[dict] = batch["input_cameras"]
        target_cameras: list[dict] = batch["target_cameras"]
        target_ids: list[int] = batch["target_ids"]

        num_inputs = len(input_images)
        num_targets = len(target_ids)

        # ---- Prepare image tensors ----
        imgs_tensor = self._load_images(input_images)  # [num_inputs, 3, H, W]

        # ---- Prepare camera tensors ----
        all_cameras = input_cameras + target_cameras
        extrinsics = self._build_extrinsics(all_cameras)  # [V, 4, 4]
        intrinsics = self._build_intrinsics(all_cameras)   # [V, 3, 3]

        # ---- Build model input ----
        # Input views: conditioning images; target views: pure noise.
        init_noise = self.flow_matching.noise_scale * torch.randn(
            num_targets, 3, self.img_size, self.img_size,
            device=self.device, dtype=self.dtype,
        )
        x = torch.cat([imgs_tensor, init_noise], dim=0)  # [V, 3, H, W]

        # noisy_mask: True for target views (views to be denoised)
        V = num_inputs + num_targets
        noisy_mask = torch.zeros(V, dtype=torch.bool, device=self.device)
        noisy_mask[num_inputs:] = True

        # Add batch dimension → [1, V, ...]
        x = x.unsqueeze(0)
        extrinsics = extrinsics.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        intrinsics = intrinsics.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        noisy_mask = noisy_mask.unsqueeze(0)

        # ---- Sampling ----
        with torch.no_grad():
            samples = self._sample_scene(x, extrinsics, intrinsics, noisy_mask)

        # ---- Extract and convert target views ----
        target_samples = samples[0, num_inputs:]        # [num_targets, 3, H, W]
        target_samples = (target_samples.clamp(-1, 1) + 1) / 2  # → [0, 1]
        target_samples = target_samples.float().cpu()

        predictions: dict[int, Image.Image] = {}
        for i, target_id in enumerate(target_ids):
            img_np = (target_samples[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            predictions[target_id] = Image.fromarray(img_np)

        return predictions

    # ------------------------------------------------------------------ #
    # Sampling                                                             #
    # ------------------------------------------------------------------ #

    def _sample_scene(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        noisy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Euler ODE integration from t=1 to t=0 on the target views.

        Args:
            x:           [1, V, 3, H, W] — conditioning views real, target views = noise.
            extrinsics:  [1, V, 4, 4]   — w2c camera matrices.
            intrinsics:  [1, V, 3, 3]   — 3×3 K matrices.
            noisy_mask:  [1, V]         — True for target views.

        Returns:
            Denoised output [1, V, 3, H, W] in [-1, 1].
        """
        B = x.shape[0]
        mask_broad = noisy_mask[:, :, None, None, None].float().to(x.dtype)

        z = x.clone()  # target views already initialised to noise
        t_cur = torch.ones(B, device=self.device, dtype=self.dtype)
        dt = 1.0 / self.num_sampling_steps
        t_eps = self.flow_matching.t_eps
        pred_mode = self.flow_matching.pred_mode

        for _ in range(self.num_sampling_steps):
            out = self.model(z, t_cur, extrinsics, intrinsics, noisy_mask)

            if pred_mode == "x0_to_v":
                t_broad = t_cur.reshape(-1, 1, 1, 1, 1).clamp_min(t_eps)
                v = (z - out) / t_broad
            else:
                v = out

            z = z - dt * v
            t_cur = t_cur - dt
            # Restore conditioning views at every step
            z = mask_broad * z + (1 - mask_broad) * x

        return z

    # ------------------------------------------------------------------ #
    # Camera helpers                                                       #
    # ------------------------------------------------------------------ #

    def _build_extrinsics(self, cameras: list[dict]) -> torch.Tensor:
        """Convert c2w transform_matrices to w2c for the model.

        eval_bench camera records use opengl_c2w convention; the EasyLab-JiT
        DL3DV dataset stores world-to-camera matrices, so we invert here.
        """
        exts = []
        for cam in cameras:
            c2w = torch.tensor(cam["transform_matrix"], dtype=torch.float32)
            w2c = torch.linalg.inv(c2w)
            exts.append(w2c)
        return torch.stack(exts)  # [V, 4, 4]

    def _build_intrinsics(self, cameras: list[dict]) -> torch.Tensor:
        """Build 3×3 K matrices, scaled to self.img_size."""
        Ks = []
        for cam in cameras:
            intr = cam.get("intrinsics") or {}
            orig_w = float(intr.get("width") or self.img_size)
            orig_h = float(intr.get("height") or self.img_size)
            fl_x = float(intr.get("fl_x") or (orig_w / 2.0))
            fl_y = float(intr.get("fl_y") or (orig_h / 2.0))
            cx = float(intr.get("cx") or (orig_w / 2.0))
            cy = float(intr.get("cy") or (orig_h / 2.0))

            sx = self.img_size / orig_w
            sy = self.img_size / orig_h

            K = torch.tensor(
                [
                    [fl_x * sx, 0.0,        cx * sx],
                    [0.0,       fl_y * sy,  cy * sy],
                    [0.0,       0.0,        1.0],
                ],
                dtype=torch.float32,
            )
            Ks.append(K)
        return torch.stack(Ks)  # [V, 3, 3]

    # ------------------------------------------------------------------ #
    # Image helpers                                                        #
    # ------------------------------------------------------------------ #

    def _load_images(self, pil_images: list[Image.Image]) -> torch.Tensor:
        """Resize PIL images to img_size and normalise to [-1, 1]."""
        tensors = []
        for img in pil_images:
            img = img.convert("RGB").resize(
                (self.img_size, self.img_size), Image.BILINEAR
            )
            arr = np.array(img).astype(np.float32)
            t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            t = t / 255.0 * 2.0 - 1.0
            tensors.append(t)
        return torch.stack(tensors).to(device=self.device, dtype=self.dtype)
