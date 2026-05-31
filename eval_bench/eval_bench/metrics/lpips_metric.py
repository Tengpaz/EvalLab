"""Optional LPIPS metric wrapper."""

from __future__ import annotations

from eval_bench.io.images import ImageData


class LPIPSMetric:
    """LPIPS wrapper initialized lazily when torch/lpips are available."""

    def __init__(self, net: str = "alex", device: str = "cuda:0") -> None:
        try:
            import lpips  # type: ignore
            import numpy as np  # type: ignore
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError(
                "LPIPS requires optional packages: torch, lpips, numpy. "
                "Install them in the evaluation environment or remove lpips from metrics.enabled."
            ) from exc
        self.lpips = lpips
        self.np = np
        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()

    def _tensor(self, image: ImageData):
        arr = self.np.array(image.pixels, dtype=self.np.float32).reshape(image.height, image.width, 3)
        arr = arr / 127.5 - 1.0
        arr = self.np.transpose(arr, (2, 0, 1))[None, ...]
        return self.torch.from_numpy(arr).to(self.device)

    def __call__(self, pred: ImageData, gt: ImageData) -> float:
        with self.torch.no_grad():
            score = self.model(self._tensor(pred), self._tensor(gt))
        return float(score.detach().cpu().item())
