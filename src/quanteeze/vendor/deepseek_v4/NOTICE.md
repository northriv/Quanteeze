# DeepSeek V4 encoding (third party, MIT)

`encoding_dsv4.py` and `tests/` are DeepSeek's, unmodified, from
deepseek-ai/DeepSeek-V4-Flash-0731 on Hugging Face:

- `encoding/encoding_dsv4.py` and `encoding/tests/*` at commit
  7872f01b1d1fe23eabc4c98b48bffcef5a386062
  (https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding);
- `LICENSE`, the repository's, at the same commit.

They are under DeepSeek's MIT License in `LICENSE` (Copyright (c) 2023 DeepSeek), not
Quanteeze's dual license (`GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0`).  DeepSeek V4 ships no Jinja chat template; this file is
its prompt format and tool-call parser.  `quanteeze.dsml` loads it to prompt
DeepSeek V4 in the server and to parse its DSML tool calls.
