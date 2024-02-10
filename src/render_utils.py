# isort: skip_file
import numpy as np
import torch
import nvdiffrast.torch as dr


# ---------------------------------------------------------------------------- #
# Transformation
# ---------------------------------------------------------------------------- #
# From kaolin.render.camera
def up_to_homogeneous(vectors: torch.Tensor):
    """Up-projects vectors to homogeneous coordinates of four dimensions.
    If the vectors are already in homogeneous coordinates, this function return the inputs.

    Args:
        vectors (torch.Tensor):
            the inputs vectors to project, of shape :math:`(..., 3)`

    Returns:
        (torch.Tensor): The projected vectors, of same shape than inputs but last dim to be 4
    """
    if vectors.shape[-1] == 4:
        return vectors
    return torch.cat([vectors, torch.ones_like(vectors[..., 0:1])], dim=-1)


def to_transformation_matrix(R_3x3: torch.Tensor, t_3x1: torch.Tensor):
    """Converts rotation matrix and translation vector to transformation matrix."""
    return torch.cat(
        [
            torch.nn.functional.pad(R_3x3, (0, 0, 0, 1)),
            torch.nn.functional.pad(t_3x1, (0, 0, 0, 1), value=1),
        ],
        dim=1,
    )


# ---------------------------------------------------------------------------- #
# nvdiffrast
# ---------------------------------------------------------------------------- #
_device2glctx = {}


# From kaolin.render.mesh.rasterization
def _get_nvdiff_glctx(device):
    if device not in _device2glctx:
        _device2glctx[device] = dr.RasterizeCudaContext(device=device)
    return _device2glctx[device]


def _rasterize(
    pos_bxnx4: torch.Tensor, tri_fx3: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """A wrapper of nvdiffrast.rasterize()."""
    assert pos_bxnx4.ndim == 3 and pos_bxnx4.shape[-1] == 4, pos_bxnx4.shape
    assert tri_fx3.ndim == 2 and tri_fx3.shape[-1] == 3, tri_fx3.shape
    glctx = _get_nvdiff_glctx(pos_bxnx4.device)

    # The last dimension is triangle_id (1-indexed). 0 for empty space.
    rast_bxhxwx4, rast_db = dr.rasterize(
        glctx, pos_bxnx4.contiguous(), tri_fx3.contiguous(), (height, width)
    )
    return rast_bxhxwx4, rast_db


def to_camera_and_ndc(
    pos_nx3: torch.Tensor, mv_bx4x4: torch.Tensor, proj_bx4x4: torch.Tensor
):
    """Transforms vertices from world space to camera space and NDC.

    Args:
        pos_nx3 (torch.Tensor):
            the vertices to transform, of shape :math:`(n, 3)`
        mv_bx4x4 (torch.Tensor):
            the model view matrix, of shape :math:`(b, 4, 4)`
        proj_bx4x4 (torch.Tensor):
            the camera projection matrix, of shape :math:`(b, 4, 4)`

    Returns:
        (torch.Tensor, torch.Tensor):
            the camera space vertices and NDC vertices, of shape :math:`(b, n, 4)`
    """
    pos_nx4 = up_to_homogeneous(pos_nx3)
    pos_bxnx4_cam = pos_nx4 @ mv_bx4x4.transpose(-1, -2)
    assert pos_bxnx4_cam.ndim == 3, pos_bxnx4_cam.shape
    pos_bxnx4_clip = pos_bxnx4_cam @ proj_bx4x4.transpose(-1, -2)
    return pos_bxnx4_cam, pos_bxnx4_clip


def rasterize(
    pos_nx3: torch.Tensor,
    tri_fx3: torch.Tensor,
    mv_bx4x4: torch.Tensor,
    proj_bx4x4: torch.Tensor,
    height: int,
    width: int,
    flip_y=True,
    return_antialias_mask=False,
):
    """Rasterize a mesh.

    Args:
        pos_nx3 (torch.Tensor):
            the vertices to rasterize, of shape :math:`(n, 3)`
        tri_fx3 (torch.Tensor):
            the triangle indices, of shape :math:`(f, 3)`
        mv_bx4x4 (torch.Tensor):
            the model view matrix, of shape :math:`(b, 4, 4)`
        proj_bx4x4 (torch.Tensor):
            the camera projection matrix, of shape :math:`(b, 4, 4)`
        height (int):
            the height of the image
        width (int):
            the width of the image
        flip_y (bool):
            whether to flip the image vertically. The memory order of image data in OpenGL convention is bottom-up.
        return_antialias_mask (bool):
            whether to return the antialias mask

    Returns:
        torch.Tensor: the rasterized image, of shape :math:`(b, h, w, 4)`
        torch.Tensor: the image-space derivatives, of shape :math:`(b, h, w, 4)`
        torch.Tensor: (Optional) the antialias mask if `return_antialias_mask` is True, of shape :math:`(b, h, w, 1)`
    """
    _, pos_bxnx4_clip = to_camera_and_ndc(pos_nx3, mv_bx4x4, proj_bx4x4)
    rast_bxhxwx4, rast_db = _rasterize(pos_bxnx4_clip, tri_fx3, height, width)

    if return_antialias_mask:
        hard_mask = torch.clamp(rast_bxhxwx4[..., -1:], 0, 1)
        # [b, h, w, 1]
        antialias_mask = dr.antialias(hard_mask, rast_bxhxwx4, pos_bxnx4_clip, tri_fx3)

    # The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
    if flip_y:
        rast_bxhxwx4 = torch.flip(rast_bxhxwx4, dims=[1])
        rast_db = torch.flip(rast_db, dims=[1])
        if return_antialias_mask:
            antialias_mask = torch.flip(antialias_mask, dims=[1])

    if return_antialias_mask:
        return rast_bxhxwx4, rast_db, antialias_mask
    return rast_bxhxwx4, rast_db


def map_uv(
    uv_mx2: torch.Tensor,
    uv_idx_fx3: torch.Tensor,
    rast_bxhxwx4: torch.Tensor,
    rast_db: torch.Tensor,
    flip_v=True,
):
    """Map uv coordinates to the image space.

    Args:
        uv_mx2 (torch.Tensor):
            the uv coordinates, of shape :math:`(m, 2)`
        uv_idx_fx3 (torch.Tensor):
            the face uv indices, of shape :math:`(f, 3)`
        rast_bxhxwx4 (torch.Tensor):
            the rasterized image, of shape :math:`(b, h, w, 4)`
        rast_db (torch.Tensor):
            the image-space derivatives, of shape :math:`(b, h, w, 4)`
        flip_v (bool):
            whether to flip v. The memory order of image data in OpenGL convention is bottom-up.

    Returns:
        (torch.Tensor, torch.Tensor):
            the mapped uv coordinates and the image-space derivatives, of shape :math:`(b, h, w, 2)`
    """
    uv_map_bxhxwx2, attrs_db = dr.interpolate(
        uv_mx2.contiguous(),
        rast_bxhxwx4.contiguous(),
        uv_idx_fx3.contiguous(),
        rast_db=rast_db,
        diff_attrs="all",
    )
    if flip_v:
        uv_map_bxhxwx2 = torch.stack(
            (uv_map_bxhxwx2[..., 0], 1 - uv_map_bxhxwx2[..., 1]), dim=-1
        )
    return uv_map_bxhxwx2, attrs_db


def map_tex(
    tex_HxWx3: torch.Tensor, uv_map_bxhxwx2: torch.Tensor, attrs_db, max_mip_level=None
):
    """Texture mapping.

    Args:
        tex_HxWx3 (torch.Tensor):
            the texture, of shape :math:`(H, W, 3)`
        uv_map_bxhxwx2 (torch.Tensor):
            the mapped uv coordinates, of shape :math:`(b, h, w, 2)`
        attrs_db (torch.Tensor):
            the image-space derivatives of uv map, of shape :math:`(b, h, w, 2)`
        max_mip_level (int):
            the max number of mipmaps

    Returns:
        torch.Tensor:
            the mapped texture, of shape :math:`(b, h, w, 3)`
    """
    tex_bxHxWx3 = tex_HxWx3.unsqueeze(0).tile(uv_map_bxhxwx2.shape[0], 1, 1, 1)
    img_bxhxwx3 = dr.texture(
        tex_bxHxWx3.contiguous(),
        uv_map_bxhxwx2.contiguous(),
        attrs_db.contiguous(),
        filter_mode="linear-mipmap-linear",
        max_mip_level=max_mip_level,
    )
    return img_bxhxwx3


def map_multi_tex(
    tex_KxHxWx3: torch.Tensor,
    im_material_idx_bxhxw: torch.Tensor,
    uv_map_bxhxwx2: torch.Tensor,
    attrs_db,
    max_mip_level=None,
):
    """Texture mapping (multiple textures).

    Args:
        tex_KxHxWx3 (torch.Tensor):
            multiple textures, of shape :math:`(K, H, W, 3)`
        im_material_idx_bxhxw (torch.Tensor):
            the material index of each image, of shape :math:`(b, h, w)`
        uv_map_bxhxwx2 (torch.Tensor):
            the mapped uv coordinates, of shape :math:`(b, h, w, 2)`
        attrs_db (torch.Tensor):
            the image-space derivatives of uv map, of shape :math:`(b, h, w, 2)`
        max_mip_level (int):
            the max number of mipmaps

    Returns:
        torch.Tensor:
            the mapped texture, of shape :math:`(b, h, w, 3)`
    """
    img_bxhxwx3 = []
    for i in range(tex_KxHxWx3.shape[0]):
        tex_i_HxWx3 = tex_KxHxWx3[i]
        tex_bxHxWx3 = tex_i_HxWx3.unsqueeze(0).tile(uv_map_bxhxwx2.shape[0], 1, 1, 1)
        _img_bxhxwx3 = dr.texture(
            tex_bxHxWx3.contiguous(),
            uv_map_bxhxwx2.contiguous(),
            attrs_db.contiguous(),
            filter_mode="linear-mipmap-linear",
            max_mip_level=max_mip_level,
        )
        mask = im_material_idx_bxhxw == i
        img_bxhxwx3.append(_img_bxhxwx3 * mask.unsqueeze(-1))
    img_bxhxwx3 = torch.stack(img_bxhxwx3, dim=0).sum(0)
    return img_bxhxwx3


def render_images(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    face_uvs_idx: torch.Tensor,
    tex: torch.Tensor,
    view_matrix: torch.Tensor,
    projection_matrix: torch.Tensor,
    height: int,
    width: int,
    model_matrix: torch.Tensor = None,
    bg_color: float = 1.0,
    max_mip_level: int = None,
):
    """Render images.

    Args:
        vertices (torch.Tensor):
            the vertices to rasterize, of shape :math:`(n, 3)`
        faces (torch.Tensor):
            the triangle indices, of shape :math:`(f, 3)`
        uvs (torch.Tensor):
            the uv coordinates, of shape :math:`(f, 3)`
        face_uvs_idx (torch.Tensor):
            the uv indices, of shape :math:`(f, 3)`
        tex (torch.Tensor):
            the texture, of shape :math:`(H, W, 3)`
        view_matrix (torch.Tensor):
            the view matrix, of shape :math:`(b, 4, 4)`
        projection_matrix (torch.Tensor):
            the projection matrix, of shape :math:`(b, 4, 4)`
        height (int):
            the height of the image
        width (int):
            the width of the image
        model_matrix (torch.Tensor):
            (Optional) the model matrix, of shape :math:`(4, 4)` or :math:`(b, 4, 4)`
        bg_color (float):
            (Optional) the background color
        max_mip_level (int):
            (Optional) the max number of mipmaps

    Returns:
        torch.Tensor:
            the rendered images in [0, 1], of shape :math:`(b, h, w, 3)`
        torch.Tensor:
            the boolean visibility mask, of shape :math:`(b, h, w)`
    """
    if model_matrix is not None:
        model_view_matrix = view_matrix @ model_matrix
    else:
        model_view_matrix = view_matrix
    rast, rast_db = rasterize(
        vertices, faces, model_view_matrix, projection_matrix, height, width
    )
    uv_map, attrs_db = map_uv(uvs, face_uvs_idx, rast, rast_db)
    imgs = map_tex(tex, uv_map, attrs_db, max_mip_level=max_mip_level)
    face_idx = rast[..., -1].long()
    imgs[face_idx == 0] = bg_color
    return imgs, face_idx > 0


# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #
@torch.no_grad()
def to_uint8(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.clip(x * 255, 0, 255).astype(np.uint8)
