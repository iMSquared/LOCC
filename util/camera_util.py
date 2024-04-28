import jax.numpy as jnp
import numpy as np
import util.transform_util as tutil


def pixel_ray(cam_pos, cam_quat, intrinsic, near, far):
    pixel_size = intrinsic[:2]
    cam_zeta = intrinsic[2:]

    K_mat = jnp.array([[cam_zeta[0], 0, cam_zeta[2]],
                    [0, cam_zeta[1], cam_zeta[3]],
                    [0,0,1]])

    # pixel= PVM (colomn-wise)
    # M : points
    # V : inv(cam_SE3)
    # P : Z projection and intrinsic matrix  
    x_grid_idx, y_grid_idx = jnp.meshgrid(jnp.arange(pixel_size[1])[::-1], jnp.arange(pixel_size[0])[::-1])
    pixel_pnts = jnp.concatenate([x_grid_idx[...,None], y_grid_idx[...,None], jnp.ones_like(y_grid_idx[...,None])], axis=-1)
    pixel_pnts = jnp.array(pixel_pnts, dtype=jnp.float32)
    K_mat_inv = jnp.linalg.inv(K_mat)
    pixel_pnts = jnp.matmul(K_mat_inv,pixel_pnts[...,None])[...,0]
    rays_s_canonical = pixel_pnts * near
    rays_e_canonical = pixel_pnts * far

    # cam SE3 transformation
    rays_s = tutil.pq_action(cam_pos, cam_quat, rays_s_canonical)
    rays_e = tutil.pq_action(cam_pos, cam_quat, rays_e_canonical)
    ray_dir = rays_e - rays_s
    ray_dir_normalized = tutil.normalize(ray_dir)

    return rays_s, rays_e, ray_dir_normalized