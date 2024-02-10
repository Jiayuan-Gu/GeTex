# Get Textures for CAD from Images

## Installation

```bash
conda create -n getex python=3.9
pip install -r requirements.txt
```

Tested under `torch==2.1.2`, `nvdiffrast==0.3.1`.

If you need to annotate foreground masks by SAM, please install `segment-anything` as `pip install git+https://github.com/facebookresearch/segment-anything.git`.

## HowTo

### Bake the texture of a single object from a single image

```bash
python bake_object_texture.py
```
