# cuttings-model-v1

A trained encoder for drill cuttings photographs.

```bash
uv sync
cp .env.example .env    # CUTTINGS_CHECKPOINT=/path/to/ssl24_geochem_fusion_backbone.ckpt
uv run jupyter lab notebooks/validate.ipynb
```

```python
backbone = load_backbone(Path("ssl24_geochem_fusion_backbone.ckpt"))
# or off the shelf: build("dinov2_vitb14", pretrained=True)
frame = read_cr2_frame(Path("sample/....CR2"))     # 4000x6000 uint8
origins = lattice_origins(frame)                   # 442 crops of 224 px
batch = to_batch(frame, origins=origins)
vector = descriptor(crop_features(backbone, batch, device=device))
```
