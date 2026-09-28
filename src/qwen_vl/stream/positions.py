"""Independent logical-token and multimodal-coordinate cursors."""
from dataclasses import dataclass
import torch


@dataclass
class PositionLedger:
    logical_token_count: int = 0
    mrope_cursor: int = 0

    def append(self, native_model, input_ids, mm_token_type_ids, image_grid_thw=None):
        local, _ = native_model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
        )
        mrope = local + self.mrope_cursor
        length = input_ids.shape[1]
        logical = torch.arange(
            self.logical_token_count,
            self.logical_token_count + length,
            device=input_ids.device,
        ).reshape(1, 1, -1)
        self.logical_token_count += length
        self.mrope_cursor = int(mrope.max()) + 1
        return torch.cat((logical, mrope), dim=0)
