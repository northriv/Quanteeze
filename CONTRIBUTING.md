# Contributing to Quanteeze

Thank you for your interest in Quanteeze. Bug reports, benchmark results on your hardware, and pull requests are all welcome.

Quanteeze is developed and maintained by one person, so responses may take some time.

## Reporting a bug

Please open an issue with:

- your machine (Mac model and memory, or GPU, VRAM, and host RAM), OS version, and the Quanteeze commit (`git rev-parse --short HEAD`);
- the model checkpoint and any `QUANTEEZE_*` settings you changed;
- what you ran, what you expected, and what happened, including the relevant part of the server log.

Please do not attach model weights, private prompts, or other data you are not free to share.

## Pull requests

### Contributor License Agreement (required)

Before a pull request can be merged, you must accept the [Contributor License Agreement](CLA.md) by commenting on your pull request with:

> I have read the Quanteeze Individual Contributor License Agreement (CLA.md, version 1.0) and I hereby accept it.

Why: Quanteeze is offered under `GPL-2.0-only OR PolyForm Shield 1.0.0`, and also under other terms (for example, to companies that integrate it into products). The CLA lets the maintainer keep offering your contribution under all of these terms. You keep the copyright in your contribution. You only need to accept once; it covers your later pull requests too.

Pull requests without an accepted CLA cannot be merged. The maintainer may instead fix the reported problem independently.

### Before you start

For anything larger than a small fix, please open an issue first to discuss the change. Some directions may already be in progress or out of scope, and the maintainer may not be able to explain why in detail.

### Guidelines

- Keep each pull request focused on one change.
- Add or update tests for the change, and run the related test files:

  ```bash
  uv sync
  .venv/bin/python -m pytest tests/<related test file> -q
  ```

- Keep outputs bit-identical to the reference model when masks are off and standard precision is used. Custom Metal and CUDA kernels must keep checking their first calls against the reference implementation.
- Register new environment variables in `src/quanteeze/variables.py` and read them through `variables.get()` and its helpers, not `os.environ`.
- Do not fork `mlx-lm` or `mlx-vlm`; use the hooks under `src/quanteeze/arch/`.
- Code, docstrings, and commit messages are in English.
- If your pull request includes code from another project, state its source and license. Code under licenses incompatible with `GPL-2.0-only` cannot be accepted.

## License

By contributing, you agree that your contributions are licensed as described in the [CLA](CLA.md) and the [LICENSE](LICENSE) file.
