# DeepSeek V4.1 encoding (third party, MIT)

`encoding.py` and `tests/` are DeepSeek's, unmodified, from
deepseek-ai/DeepSeek-V4.1-Flash-1017 on Hugging Face:

- `encoding/encoding.py` and `encoding/tests/*` at commit
  1bae0e1088993b9b82de08f59de5be5a34e42170
  (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding);
- `LICENSE`, the repository's, at the same commit.

They are under DeepSeek's MIT License in `LICENSE` (Copyright (c) 2023 DeepSeek), not
Quanteeze's dual license (`GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0`).  DeepSeek V4.1 ships no Jinja chat template, and its
format is not V4's: its DSML tags have a leading space (`<｜DSML｜ calls>`), its
reasoning effort is a budget from 1 to 100, and a system message may come
mid-conversation.  `quanteeze.dsml` loads this file to prompt DeepSeek V4.1 in the
server and to parse its DSML tool calls.  Quantized copies of the model
(mlx-community's among them) leave the `encoding/` folder out; this copy is
used for them.
