# Get Textures for CAD from Images

## Installation

```bash
conda create -n getex python=3.9
pip install -r requirements.txt
```

Tested under `torch==2.1.2`, `nvdiffrast==0.3.1`, `trimesh==4.1.3`.

If you need to annotate foreground masks by SAM, please install "segment-anything": `pip install git+https://github.com/facebookresearch/segment-anything.git`.

## HowTo

### Bake the texture of a single object from a single image

```bash
python bake_object_texture.py -h
#  --mesh_path str       path to the object mesh (default: assets/objects/opened_pepsi_can/textured.dae)
#  --ref_image_path str  path to the reference image (default: assets/ref_images/move_near_real_1_2.sd-x4-43.png)
#  --model_matrix ndarray
#                        model matrix, [4, 4] (default: [[ 9.9999994e-01 -5.6162389e-06 4.1270844e-04 -3.8991699e-01] [ 4.1270826e-04 -2.5391579e-05 -9.9999988e-01 2.0712100e-01] [ 5.6267454e-06 1.0000000e+00 -2.5391579e-05 9.5182002e-01] [ 0.0000000e+00 0.0000000e+00
#                        0.0000000e+00 1.0000000e+00]])
#  --view_matrix ndarray
#                        view matrix, [4, 4] (default: [[ 0.093 0.996 -0. -0.242] [-0.704 0.066 0.707 -0.785] [ 0.704 -0.066 0.707 -1.043] [ 0. 0. 0. 1. ]])
#  --fov_x ndarray       field of view in radians (default: 1.2363)
#  --ref_mask_path str   path to the reference mask. If specified but not found, an interactive annotation will be performed. (default: assets/ref_masks/mask_pepsi_can.png)
#  --output_path str     output path for the baked mesh (default: None)
```

The user should have obtained a roughly correct geometry with reasonable UV maps (`mesh_path`) before using the script. The camera instrinsic parameters (`fov_x`) should also be known.

The user needs to first manually align the object with the reference image (`ref_image_path`) to obtain `model_matrix` given a known `view_matrix`. Note that the only thing we care about is `model_view_matrix` instead of individual ones, and thus you can fix `view_matrix` and tune `model_matrix`. If a foreground mask of the object in the reference image (`ref_mask_path`) is provided, we will first refine the `model_matrix` by matching the silhouette via differentiable rendering. Next, we will optimize the texture by matching the rendered RGB image and renference image via differentiable rendering.

We use `simple-parsing` to define program arguments by `dataclass`, and also support reading from a config file (json or yaml) via `--config_path={CONFIG_FILE}`.

<p align="center">
    <img src="assets/ref_images/move_near_real_1_2.sd-x4-43.png" width="24%" alt="reference image" />
    <img src="assets/ref_masks/mask_pepsi_can.png" width="24%" alt="foreground mask of pepsi can" />
    <img src="static/img/textured.optimized_image.png" width="24%" alt="optimized rendering result" />
    <img src="static/img/partially_baked.gif" width="24%" alt="animation for partially baked mesh" />
</p>

> From left to right: reference image, foreground mask, rendered image of optimized texture, animation for partially baked mesh.

### Complete the texture from a partially baked mesh

The previous step can only generate a partially baked mesh. We will use [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) to hallucinate multi-view images. First, you need to install extra dependencies: `pip install -r requirements-extra.txt`.

```bash
python complete_object_texture.py -h
#  --mesh_path str       path to the object mesh (partially baked) (default: assets/objects/opened_pepsi_can/textured.baked.glb)
#  --delta_R ndarray     extra rotation to rotate the object to a canonical pose, [3, 3]. Require manually tuning. (default: [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]])
#  --model_rotation ndarray
#                        model matrix to render the object (at the canonical pose) as the image condition for Zero123++, [4, 4] (default: [[ 0.8575973 0.5 0.12052744] [-0.49513403 0.8660254 -0.06958655] [-0.1391731 0. 0.99026807]])
#  --seed int            random seed for Zero123++. Require manually tuning. (default: 25)
#  --output_path str     output path for the baked mesh (default: None)
#  --force_generate bool, --noforce_generate bool
#                        force to re-generate zero123++ outputs (default: False)
```

<p align="center">
  <img src="static/img/textured.baked_texture.png" width="24%" alt="partially baked texture" />
  <img src="static/img/textured.baked.zero123plus.png" width="16%" alt="zero123++ output" />
  <img src="static/img/textured.baked.completed_texture.png" width="24%" alt="completed texture" />
  <img src="static/img/completed_texture.gif" width="24%" alt="animation of completed mesh" />
</p>

> From left to right: partially baked texture, zero123++ output, completed texture, animation for completed mesh.
