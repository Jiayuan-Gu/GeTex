from dataclasses import dataclass
from pathlib import Path

import calibur
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from trimesh.visual.material import PBRMaterial

from src import render_utils

# From kaolin.render.camera.intrinsics_pinhole
# Default near / far values are best used for small / medium scale scenes.
# Use cases with bigger scale should use other explicit values.
DEFAULT_NEAR = 1e-2
DEFAULT_FAR = 1e2


def complete_texture(
    mesh_path,
    delta_R=None,
    model_rotation=None,
    force_generate=False,
    seed=42,
    output_path=None,
):
    """Complete the texture of an object mesh.

    Args:
        mesh_path (str):
            the path to the object mesh (partially baked)
        delta_R (np.ndarray):
            the extra rotation to rotate the object to a canonical pose. Zero123++ defines its camera elevations based on a canonical pose.
        model_rotation (np.ndarray):
            the model rotation to render an image condition for Zero123++ to generate multi-view images for optimization.
        force_generate (bool):
            whether to force generate multi-view images by Zero123++
        seed (int):
            the seed for Zero123++
        output_path (str):
            (Optional) the path to the output mesh. If None, the default output path (`{mesh_path_stem}.emissive.glb`) will be used.
    """
    # Load mesh
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    uvs = torch.from_numpy(mesh.visual.uv).float().cuda()
    face_uvs_idx = faces.clone().detach().cuda()

    # Load texture
    if mesh.visual.material.baseColorTexture is not None:
        tex = np.array(mesh.visual.material.baseColorTexture) / 255
    elif mesh.visual.material.emissiveTexture is not None:
        tex = np.array(mesh.visual.material.emissiveTexture) / 255
    else:
        raise ValueError("No texture found")
    plt.imshow(tex)
    plt.title("Initial texture")
    plt.show()
    tex = torch.from_numpy(tex).float().cuda()

    # NOTE: Modifying vertices should be fine as UVs are kept.
    # Normalize shape. Assume the shape is axis-aligned.
    center = (vertices.max(dim=0)[0] + vertices.min(dim=0)[0]) / 2
    vertices = vertices - center
    bbox_size = vertices.max(dim=0)[0] - vertices.min(dim=0)[0]
    vertices = vertices * (0.9 / bbox_size.max())
    print("center", center, "bbox_size", bbox_size)

    # NOTE: Require manual tuning
    # Define a canonical pose
    if delta_R is not None:
        vertices = vertices @ vertices.new_tensor(delta_R).T

    # Define a camera to render an image from the similar view when baking the texture
    view_matrix0 = render_utils.look_at(
        eye=[1.5, 0.0, 0.0],
        at=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0],
    )
    # Hardcode according to zero-123-plus training data
    cam_height0, cam_width0 = 512, 512
    focal0 = calibur.projection.fov_to_focal(np.deg2rad(47.1), cam_width0)
    projection_matrix0 = calibur.projection_gl_persp(
        cam_width0,
        cam_height0,
        cam_width0 // 2,
        cam_height0 // 2,
        focal0,
        focal0,
        DEFAULT_NEAR,
        DEFAULT_FAR,
    )
    view_matrix0 = torch.from_numpy(view_matrix0).float().unsqueeze(0).cuda()
    projection_matrix0 = (
        torch.from_numpy(projection_matrix0).float().unsqueeze(0).cuda()
    )

    # Note that we adjust model matrix instead of view matrix here.
    if model_rotation is None:
        model_matrix0 = None
    else:
        model_matrix0 = render_utils.to_transformation_matrix(
            vertices.new_tensor(model_rotation), None
        )

    # Render a view that we want to keep its original appearance
    with torch.no_grad():
        init_images, init_masks = render_utils.render_images(
            vertices,
            faces,
            uvs,
            face_uvs_idx,
            tex,
            view_matrix=view_matrix0,
            projection_matrix=projection_matrix0,
            height=cam_height0,
            width=cam_width0,
            model_matrix=model_matrix0,
        )
    init_image_pil = Image.fromarray(render_utils.to_uint8(init_images[0])).convert(
        "RGB"
    )
    plt.imshow(init_image_pil)
    plt.title("Initial image")
    plt.show()

    # ---------------------------------------------------------------------------- #
    # Zero123Plus pipeline
    # ---------------------------------------------------------------------------- #
    zero123plus_output_path = Path(mesh_path).with_suffix(".zero123plus.png")
    if not zero123plus_output_path.exists() or force_generate:
        # https://github.com/SUDO-AI-3D/zero123plus?tab=readme-ov-file#get-started

        # Load the pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )

        # Feel free to tune the scheduler!
        # `timestep_spacing` parameter is not supported in older versions of `diffusers`
        # so there may be performance degradations
        # We recommend using `diffusers==0.20.2`
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
        pipeline.to("cuda")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Run the pipeline!
        result = pipeline(
            init_image_pil, num_inference_steps=75, generator=generator
        ).images[0]
        # for general real and synthetic images of general objects
        # usually it is enough to have around 28 inference steps
        # for images with delicate details like faces (real or anime)
        # you may need 75-100 steps for the details to construct

        result.save(zero123plus_output_path)
    else:
        result = Image.open(zero123plus_output_path).convert("RGB")

    plt.imshow(result)
    plt.title("Zero123++ output")
    plt.show()

    generated_images = np.array(result).reshape((3, 320, 2, 320, 3))
    generated_images = np.transpose(generated_images, (0, 2, 1, 3, 4)).reshape(
        -1, 320, 320, 3
    )

    zero123plus_mask_path = Path(mesh_path).with_suffix(".zero123plus_mask.png")
    if not zero123plus_mask_path.exists() or force_generate:
        import rembg

        generated_masks = []
        for i in range(generated_images.shape[0]):
            _mask = rembg.remove(generated_images[i], only_mask=True)  # [H, W]
            generated_masks.append(_mask)
        generated_masks = np.stack(generated_masks)  # [B, H, W], uint8
        Image.fromarray(generated_masks.reshape(-1, 320)).save(zero123plus_mask_path)
    else:
        generated_masks = np.array(Image.open(zero123plus_mask_path).convert("L"))
        generated_masks = generated_masks.reshape(-1, 320, 320)

    plt.imshow(
        generated_masks.reshape(3, 2, 320, 320)
        .transpose(0, 2, 1, 3)
        .reshape(320 * 3, 320 * 2)
    )
    plt.title("Zero123++ mask")
    plt.show()

    # ---------------------------------------------------------------------------- #
    # Multi-view optimization
    # ---------------------------------------------------------------------------- #
    # Predefined camera poses of zero123++
    azimuths = [30, 90, 150, 210, 270, 330]
    elevations = [30, -20, 30, -20, 30, -20]
    view_matrices = []
    for azim, elev in zip(azimuths, elevations):
        R = Rotation.from_euler("YZ", [azim, elev], degrees=True).as_matrix()
        view_matrix = render_utils.look_at(
            eye=np.array([1.5, 0.0, 0.0]) @ R.T,
            at=np.array([0.0, 0.0, 0.0]),
            up=np.array([0.0, 1.0, 0.0]),
        )
        view_matrices.append(view_matrix)
    view_matrices = torch.from_numpy(np.stack(view_matrices)).float().cuda()

    cam_width1, cam_height1 = 320, 320
    focal1 = calibur.projection.fov_to_focal(np.deg2rad(47.1), cam_width1)
    projection_matrix1 = calibur.projection_gl_persp(
        cam_width1,
        cam_height1,
        cam_width1 // 2,
        cam_height1 // 2,
        focal1,
        focal1,
        DEFAULT_NEAR,
        DEFAULT_FAR,
    )
    projection_matrix1 = torch.from_numpy(projection_matrix1).float().cuda()

    # Render initial views for sanity check
    with torch.no_grad():
        imgs, _ = render_utils.render_images(
            vertices,
            faces,
            uvs,
            face_uvs_idx,
            tex,
            view_matrices,
            projection_matrix1,
            cam_height1,
            cam_width1,
        )
    plt.imshow(
        render_utils.to_uint8(
            imgs.reshape(3, 2, 320, 320, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(3 * 320, 2 * 320, 3)
        )
    )
    plt.title("Initial views")
    plt.show()

    # Prepare reference images and masks
    ref_images = torch.from_numpy(generated_images / 255).float().cuda()
    ref_masks = torch.from_numpy(generated_masks / 255).float().cuda()

    # tex_param = torch.nn.Parameter(tex.clone())
    tex_param = torch.nn.Parameter(torch.zeros_like(tex))
    optimizer = torch.optim.Adam([tex_param], lr=5e-3)

    n_iters = 200
    pbar = tqdm(total=n_iters)
    for i in range(n_iters):
        optimizer.zero_grad()

        # Optimize according to the reference images
        rendered_images, rendered_masks = render_utils.render_images(
            vertices,
            faces,
            uvs,
            face_uvs_idx,
            tex_param,
            view_matrices,
            projection_matrix1,
            cam_height1,
            cam_width1,
        )
        diff = rendered_images - ref_images
        mask = torch.logical_and(ref_masks > 0, rendered_masks)
        loss = torch.mean(diff[mask] ** 2)
        loss.backward()

        # Optimize according to the original texture
        rendered_images, rendered_masks = render_utils.render_images(
            vertices,
            faces,
            uvs,
            face_uvs_idx,
            tex_param,
            view_matrix0,
            projection_matrix0,
            cam_height0,
            cam_width0,
            model_matrix=model_matrix0,
        )
        diff = rendered_images - init_images
        mask = torch.logical_and(init_masks, rendered_masks)
        loss = torch.mean(diff[mask] ** 2)
        loss.backward()

        optimizer.step()

        pbar.update()
        pbar.set_description("loss: {:.4f}".format(loss))

    tex_pil = Image.fromarray(render_utils.to_uint8(tex_param.data))
    tex_path = Path(mesh_path).with_suffix(".completed_texture.png")
    tex_pil.save(tex_path)
    plt.imshow(tex_pil)
    plt.title("Optimized texture")
    plt.show()

    # Export texture as base color for easier visualization
    mesh.visual.material = PBRMaterial(baseColorTexture=tex_pil)
    mesh.export(Path(mesh_path).with_suffix(".albedo.glb"))
    mesh.show()

    # Export texture as emissive color to avoid being affected by lighting
    mesh.visual.material = PBRMaterial(
        emissiveTexture=tex_pil,
        baseColorFactor=(0, 0, 0),
        metallicFactor=0,
        roughnessFactor=1,
    )
    if output_path is None:
        output_path = Path(mesh_path).with_suffix(".emissive.glb")
    mesh.export(output_path)


@dataclass
class Config:
    mesh_path: str  # path to the object mesh (partially baked)
    # extra rotation to rotate the object to a canonical pose, [3, 3]. Require manually tuning.
    delta_R: np.ndarray
    # model matrix to render the object (at the canonical pose) as the image condition for Zero123++, [4, 4]
    model_rotation: np.ndarray
    seed: int = 42  # random seed for Zero123++. Require manually tuning.
    output_path: str = None  # output path for the baked mesh
    force_generate: bool = False  # force to re-generate zero123++ outputs

    def __post_init__(self):
        if self.output_path == "":
            self.output_path = None


PRESET = Config(
    mesh_path="assets/objects/opened_pepsi_can/textured.baked.glb",
    delta_R=Rotation.from_euler("y", 0, degrees=True).as_matrix(),
    model_rotation=Rotation.from_euler("yz", [8, -30], degrees=True).as_matrix(),
    seed=25,
)


def main():
    # simple-parsing supports config files
    import simple_parsing

    config = simple_parsing.parse(Config, default=PRESET, add_config_path_arg=True)
    print(config)

    complete_texture(
        config.mesh_path,
        delta_R=config.delta_R,
        model_rotation=config.model_rotation,
        force_generate=config.force_generate,
        seed=config.seed,
    )


if __name__ == "__main__":
    main()
