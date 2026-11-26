# Nemotron-Labs-Image pixel decoder (third party, NVIDIA License, non-commercial)

`l24sq.py` is derived from NVIDIA's `Nemotron-Labs-Image-Tokenizer/modeling_l24sq_vit.py` in
`nvidia/Nemotron-Labs-Image-30B-A3B` on Hugging Face, at commit
`d7c2ba6a59a384080f0bdd5b1e846da03804cdd1`
(https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B/tree/d7c2ba6a59a384080f0bdd5b1e846da03804cdd1).
That file carries:

    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    Use of this file is governed by the NVIDIA License in the repository root (LICENSE).

`LICENSE` beside this file is that repository's, at the same commit, unmodified. It governs
`l24sq.py` **solely** — NOT Quanteeze's `GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0` dual
license. Its use limitation (Section 3.3) applies: `l24sq.py`, and any derivative works thereof,
may be used non-commercially only, meaning **for research or evaluation purposes only**.
Section 3.1 requires keeping a complete unmodified copy of `LICENSE` and retaining all copyright,
patent, trademark, and attribution notices with any redistribution.

What was changed: none of NVIDIA's original Python/PyTorch code is copied verbatim; the encoder,
the Leech-lattice quantizer, and the decoder of `modeling_l24sq_vit.py` were reimplemented in MLX
(without FlashAttention or training paths) in Quanteeze on 2026-11-26 from reading that file. It
reads user-supplied weights from `Nemotron-Labs-Image-Tokenizer/model.safetensors`, which are not
distributed in this repository.

License isolation & separability:
- The core Quanteeze inference server (`quanteeze.serve`) never imports
  `src/quanteeze/vendor/nemotron_image/`.
- `quanteeze.vendor.nemotron_image.l24sq` is imported lazily only when `quanteeze.nemotron_image.draw()`
  (or `python -m quanteeze.image_mcp`) is explicitly invoked.
- Recipients redistributing or using Quanteeze under pure `GPL-2.0-only` or for commercial
  purposes must omit `src/quanteeze/vendor/nemotron_image/` (and not use the optional
  Nemotron-Labs-Image integration); the rest of Quanteeze is unaffected.

The helper modules `quanteeze.nemotron_image` (using the model card's system prompt text from its
`infer.py` example) and `quanteeze.image_mcp` (the MCP server tool) only invoke the decoder
interface and user-supplied converted checkpoints, remaining under Quanteeze's own license when
separated; however, running Nemotron-Labs-Image-30B-A3B, converted checkpoints thereof, and
generated image outputs remain subject to the NVIDIA License's use limitation (Section 3.3:
research or evaluation purposes only). No model weights (original or converted) are included in
this repository.
