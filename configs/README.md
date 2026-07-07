# Configuration Files

This directory contains YAML configuration files for the Verification Arbitrage framework.

- `default.yaml` — Development defaults (mid-cap liquidity, no MoA, three-tier risk enabled)
- Additional profiles can be added for specific experiments (e.g., `crypto.yaml`, `high-cap.yaml`)

## Usage

```bash
python -m src.async_pipeline --config configs/default.yaml
```

All config keys have runtime equivalents via CLI flags. Config files override defaults; CLI flags override config files.
