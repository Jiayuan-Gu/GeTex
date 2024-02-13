import argparse

import calibur
import imageio
import numpy as np
import torch
import trimesh

from src import render_utils

DEFAULT_CAMERA_TRAJECTORY = list(
    zip(np.linspace(0, 4 * np.pi, 180), np.linspace(0, np.pi / 3, 180))
)


def render_frames(
    mesh_path: str,
    cam_traj=DEFAULT_CAMERA_TRAJECTORY,
    height=512,
    width=640,
    fov_x=1.2363,
    scale=0.9,
    distance=1.5,
):
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    uvs = torch.from_numpy(mesh.visual.uv).float().cuda()
    face_uvs_idx = faces.clone().detach().cuda()

    # Normalize shape. Assume the shape is axis-aligned.
    center = (vertices.max(dim=0)[0] + vertices.min(dim=0)[0]) / 2
    vertices = vertices - center
    bbox_size = vertices.max(dim=0)[0] - vertices.min(dim=0)[0]
    vertices = vertices * (scale / bbox_size.max())
    print("center", center, "bbox_size", bbox_size)

    tex_pil = mesh.visual.material.baseColorTexture
    tex = torch.from_numpy(np.array(tex_pil) / 255).float().cuda()

    focal_x = calibur.projection.fov_to_focal(fov_x, width)
    projection_matrix = calibur.projection_gl_persp(
        width, height, width // 2, height // 2, focal_x, focal_x, 1e-2, 1e2
    )
    projection_matrix = torch.from_numpy(projection_matrix).float().unsqueeze(0).cuda()

    # Can be batched to accelerate
    frames = []
    for azim, elev in cam_traj:
        eye = np.array(
            [np.cos(elev) * np.sin(azim), np.sin(elev), np.cos(elev) * np.cos(azim)]
        )
        eye = eye * distance
        view_matrix = render_utils.look_at(
            eye,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        view_matrix = torch.from_numpy(view_matrix).float().unsqueeze(0).cuda()
        images, _ = render_utils.render_images(
            vertices,
            faces,
            uvs,
            face_uvs_idx,
            tex,
            view_matrix,
            projection_matrix,
            height,
            width,
        )
        frames.append(render_utils.to_uint8(images[0]))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mesh_path",
        type=str,
        required=True,
        help="input mesh path to render",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="output path (.mp4 or .gif) for animation",
    )
    args = parser.parse_args()

    frames = render_frames(args.mesh_path)
    if args.output_path.endswith(".mp4"):
        imageio.mimwrite(args.output_path, frames, fps=30)
    elif args.output_path.endswith(".gif"):
        imageio.mimsave(args.output_path, frames, fps=30, loop=0)
        try:
            import pygifsicle  # fmt: skip
            pygifsicle.optimize(args.output_path)
        except ImportError:
            print("`pygifsicle` not found.")
    else:
        raise NotImplementedError(f"output format {args.output_path} not supported")


if __name__ == "__main__":
    main()
