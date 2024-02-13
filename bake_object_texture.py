import os
from dataclasses import dataclass
from pathlib import Path

import calibur
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from src import render_utils

# From kaolin.render.camera.intrinsics_pinhole
# Default near / far values are best used for small / medium scale scenes.
# Use cases with bigger scale should use other explicit values.
DEFAULT_NEAR = 1e-2
DEFAULT_FAR = 1e2

DEFAULT_LR = 1e-2


def bake_texture(
    mesh_path: str,
    ref_image_path: str,
    model_matrix: np.ndarray,
    view_matrix: np.ndarray,
    fov_x: float,
    output_path: str = None,
    tex_height=256,
    tex_width=256,
    max_mip_level=4,
    n_iters=100,
    as_emission=False,
    ref_mask_path: str = None,
    remap_uvs_by_xatlas: bool = False,
):
    """Bake the texture of an object from a single image.

    This function optimizes the texture of the object mesh by minimizing the rendering loss with respect to the reference image.
    If the reference mask is provided, we will first optimize the model matrix to get a better alignment, and then optimize the texture.

    Args:
        mesh_path (str):
            the path to the object mesh
        ref_image_path (str):
            the path to the reference image
        model_matrix (np.ndarray):
            the model matrix
        view_matrix (np.ndarray):
            the view matrix
        fov_x (float):
            the horizontal field of view
        output_path (str):
            (Optional) the path to the output mesh. If None, the default output path (`{mesh_path}.baked.glb`) will be used.
        tex_height (int):
            the height of the texture
        tex_width (int):
            the width of the texture
        max_mip_level (int):
            the maximum mip level. Since there is only one image, it is better not to set it too high.
        n_iters (int):
            the number of iterations for optimization.
        as_emission (bool):
            whether to convert the texture to emission
        ref_mask_path (str):
            the path to the reference mask
        remap_uvs_by_xatlas (bool):
            whether to regenerate uvs by xatlas
    """
    # NOTE: trimesh may not correctly handle multiple materials.
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 1:
        print(
            "\033[93m"
            + "Multiple geometries found. `trimesh` may not handle multiple materials correctly. The known issue is that the uvs can be different each time. Use `--remap_uvs_by_xatlas=True` to make it deterministic."
            + "\033[0m"
        )
    del mesh

    # Load mesh
    mesh = trimesh.load(mesh_path, force="mesh")

    if remap_uvs_by_xatlas:
        import xatlas  # fmt: skip
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices[vmapping],
            faces=indices,
            visual=TextureVisuals(uv=uvs),
        )

    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    uvs = torch.from_numpy(mesh.visual.uv).float().cuda()
    face_uvs_idx = faces.clone().detach().cuda()

    # Load reference image
    ref_image_pil = Image.open(ref_image_path).convert("RGB")
    ref_image = torch.from_numpy(np.array(ref_image_pil) / 255).float()
    ref_image = ref_image.unsqueeze(0).cuda()  # [1, H, W, 3]

    # Initialize camera matrices
    cam_height, cam_width = ref_image.shape[1:3]
    focal_x = calibur.projection.fov_to_focal(fov_x, cam_width)
    projection_matrix = calibur.projection_gl_persp(
        cam_width,
        cam_height,
        cam_width // 2,
        cam_height // 2,
        focal_x,
        focal_x,
        DEFAULT_NEAR,
        DEFAULT_FAR,
    )

    _projection_matrix = torch.from_numpy(projection_matrix).float().cuda()
    if _projection_matrix.ndim == 2:
        _projection_matrix = _projection_matrix.unsqueeze(0)
    _view_matrix = torch.from_numpy(view_matrix).float().cuda()
    if _view_matrix.ndim == 2:
        _view_matrix = _view_matrix.unsqueeze(0)
    _model_matrix = torch.from_numpy(model_matrix).float().cuda()

    # Optimize pose by matching silhouette
    if ref_mask_path is not None:
        ref_mask_pil = Image.open(ref_mask_path).convert("L")
        ref_mask_pil = ref_mask_pil.resize((cam_width, cam_height), Image.LANCZOS)
        ref_mask = torch.from_numpy(np.array(ref_mask_pil) / 255).float()
        ref_mask = ref_mask.unsqueeze(0).unsqueeze(-1).cuda()  # [1, H, W, 1]

        # Parameterize delta pose
        dR = torch.nn.Parameter(torch.eye(3, dtype=torch.float, device="cuda"))
        dt = torch.nn.Parameter(torch.zeros([3, 1], dtype=torch.float, device="cuda"))
        optimizer = torch.optim.Adam([dR, dt], lr=DEFAULT_LR)

        pbar = tqdm(total=n_iters)
        for _ in range(n_iters):
            optimizer.zero_grad()

            dT = render_utils.to_transformation_matrix(dR, dt)
            _, _, soft_mask = render_utils.rasterize(
                vertices,
                faces,
                _view_matrix @ _model_matrix @ dT,
                _projection_matrix,
                cam_height,
                cam_width,
                return_antialias_mask=True,
            )
            loss = torch.nn.functional.mse_loss(soft_mask, ref_mask)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description(f"[Optimizing silhouette] Loss: {loss:.5f}")

            # Project parameters to SO(3)
            with torch.no_grad():
                U, _, Vh = torch.linalg.svd(dR)
            dR.data = U @ Vh

        # Update model matrix
        with torch.no_grad():
            dT = render_utils.to_transformation_matrix(dR, dt)
            _model_matrix = _model_matrix @ dT
            print("Delta pose:", dR.cpu().numpy(), dt.cpu().numpy())

        # Visualize optimized silhouette
        with torch.no_grad():
            masked_image = render_utils.to_uint8(ref_image * soft_mask)
        plt.imshow(masked_image[0])
        plt.show()
    else:
        ref_mask = torch.ones_like(ref_image[..., :1])

    # Initialize texture
    tex_param = torch.nn.Parameter(
        torch.full([tex_height, tex_width, 3], 0.0, device="cuda")
    )
    optimizer = torch.optim.Adam([tex_param], lr=DEFAULT_LR)

    pbar = tqdm(total=n_iters)
    for _ in range(n_iters):
        optimizer.zero_grad()

        rendered_image, rendered_mask = render_utils.render_images(
            vertices,
            faces,
            uvs,
            face_uvs_idx,
            tex_param,
            view_matrix=_view_matrix,
            projection_matrix=_projection_matrix,
            model_matrix=_model_matrix,
            height=cam_height,
            width=cam_width,
            max_mip_level=max_mip_level,
        )
        diff = rendered_image - ref_image  # [B, H, W, 3]
        mask = torch.logical_and(ref_mask.squeeze(-1) > 0, rendered_mask)
        loss = torch.mean(torch.square(diff[mask]))
        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_description("Loss: {:.4f}".format(loss))

    # Visualize optimized rendered image
    plt.imshow(render_utils.to_uint8(rendered_image[0]))
    plt.show()
    # Image.fromarray(render_utils.to_uint8(rendered_image[0])).save(
    #     Path(mesh_path).with_suffix(".optimized_image.png")
    # )

    # Pack texture into mesh
    tex_pil = Image.fromarray(render_utils.to_uint8(tex_param))
    tex_pil.save(Path(mesh_path).with_suffix(".baked_texture.png"))

    if as_emission:
        mesh.visual.material = PBRMaterial(
            emissiveTexture=tex_pil,
            baseColorFactor=(0, 0, 0),
        )
    else:
        mesh.visual.material = PBRMaterial(baseColorTexture=tex_pil)

    mesh.show()

    # Get default output path if not specified
    if output_path is None:
        _mesh_path = Path(mesh_path)
        suffix = "baked"
        output_path = _mesh_path.with_stem(_mesh_path.stem + "." + suffix)
        output_path = output_path.with_suffix(".glb")

    # Export mesh
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    print("Exporting", output_path)
    mesh.export(output_path)


@dataclass
class Config:
    mesh_path: str  # path to the object mesh
    ref_image_path: str  # path to the reference image
    model_matrix: np.ndarray  # model matrix, [4, 4]
    view_matrix: np.ndarray  # view matrix, [4, 4]
    fov_x: np.ndarray  # field of view in radians
    ref_mask_path: str  # path to the reference mask. If specified but not found, an interactive annotation will be performed.
    output_path: str = None  # output path for the baked mesh
    remap_uvs_by_xatlas: bool = False

    def __post_init__(self):
        if self.ref_mask_path == "":
            self.ref_mask_path = None
        if self.output_path == "":
            self.output_path = None


PRESET = Config(
    mesh_path="assets/objects/opened_pepsi_can/textured.dae",
    ref_image_path="assets/ref_images/move_near_real_1_2.sd-x4-43.png",
    model_matrix=np.array(
        [
            [9.9999994e-01, -5.6162389e-06, 4.1270844e-04, -3.8991699e-01],
            [4.1270826e-04, -2.5391579e-05, -9.9999988e-01, 2.0712100e-01],
            [5.6267454e-06, 1.0000000e00, -2.5391579e-05, 9.5182002e-01],
            [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
        ]
    ),
    view_matrix=np.array(
        [
            [0.093, 0.996, -0.0, -0.242],
            [-0.704, 0.066, 0.707, -0.785],
            [0.704, -0.066, 0.707, -1.043],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    fov_x=1.2363,
    ref_mask_path="assets/ref_masks/mask_pepsi_can.png",
)


def main():
    # simple-parsing supports config files
    import simple_parsing

    config = simple_parsing.parse(Config, default=PRESET, add_config_path_arg=True)
    print(config)

    ref_mask_path = config.ref_mask_path
    if ref_mask_path is not None and not os.path.exists(ref_mask_path):
        from src.sam_utils import annotate  # fmt: skip
        annotate(image_path=config.ref_image_path, output_path=ref_mask_path)

    bake_texture(
        config.mesh_path,
        config.ref_image_path,
        config.model_matrix,
        config.view_matrix,
        config.fov_x,
        ref_mask_path=ref_mask_path,
        output_path=config.output_path,
        remap_uvs_by_xatlas=config.remap_uvs_by_xatlas,
    )


if __name__ == "__main__":
    main()
