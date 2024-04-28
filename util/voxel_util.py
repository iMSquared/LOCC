import jax.numpy as jnp
import jax
import einops

import util.transform_util as tutil

def vx_intersection_idx(pos, quat, obj_vx, jkey, ns=10):
    valid_vx, xyz_vx, aabb_scaled, aabb_obj = obj_vx
    nv = valid_vx.shape[-2]
    if nv > 100:
        nv = jnp.round(nv**(1/3))
    
    pos12, quat12 = tutil.pq_multi(*tutil.pq_inv(pos[...,0,:], quat[...,0,:]), pos[...,1,:], quat[...,1,:])
    pos21, quat21 = tutil.pq_inv(pos12, quat12)
    rel_pos = jnp.stack([pos21, pos12], axis=-2)
    rel_quat = jnp.stack([quat21, quat12], axis=-2)
    xyz_vx_tf = tutil.pq_action(rel_pos[...,None,:], rel_quat[...,None,:], xyz_vx)

    aabb_int_test = aabb_obj.reshape(aabb_obj.shape[:-1] + (2,3))
    vx_scales = (aabb_scaled[...,3:] - aabb_scaled[...,:3])/nv
    vx_scales = vx_scales[...,::-1,:]
    vx_scales = jnp.stack([-vx_scales,vx_scales], axis=-2)*1.2
    aabb_int_test += vx_scales
    aabb_int_test_rev = aabb_int_test[...,::-1,:,:]

    in_pnts = jnp.all(jnp.logical_and(xyz_vx_tf > aabb_int_test_rev[..., None, 0,:], xyz_vx_tf < aabb_int_test_rev[..., None, 1,:]), axis=-1)
    
    in_pnts = jnp.logical_and(valid_vx[...,0], in_pnts)

    random_idx = jax.random.permutation(jkey, in_pnts.shape[-1])
    where_idx = jax.vmap(lambda x : jnp.where(x, size=ns, fill_value=-1))(in_pnts.reshape((-1,) + in_pnts.shape[-1:])[:,random_idx])[0]
    where_idx = where_idx.reshape(in_pnts.shape[:-1]+(ns,))
    origin_idx = jnp.concatenate([random_idx, jnp.array([-1])], axis=0)[where_idx]

    return origin_idx



def vx_intersection_mean(vx_data, pos, quat, obj_vx, jkey):
    valid_vx, xyz_vx, aabb_scaled, aabb_obj = obj_vx
    nv = valid_vx.shape[-2]
    if nv > 100:
        nv = jnp.round(nv**(1/3))
    
    pos12, quat12 = tutil.pq_multi(*tutil.pq_inv(pos[...,0,:], quat[...,0,:]), pos[...,1,:], quat[...,1,:])
    pos21, quat21 = tutil.pq_inv(pos12, quat12)
    rel_pos = jnp.stack([pos21, pos12], axis=-2)
    rel_quat = jnp.stack([quat21, quat12], axis=-2)
    xyz_vx_tf = tutil.pq_action(rel_pos[...,None,:], rel_quat[...,None,:], xyz_vx)

    aabb_int_test = aabb_obj.reshape(aabb_obj.shape[:-1] + (2,3))
    vx_scales = (aabb_scaled[...,3:] - aabb_scaled[...,:3])/nv
    vx_scales = vx_scales[...,::-1,:]
    vx_scales = jnp.stack([-vx_scales,vx_scales], axis=-2)*1.2
    aabb_int_test += vx_scales
    aabb_int_test_rev = aabb_int_test[...,::-1,:,:]

    in_pnts = jnp.all(jnp.logical_and(xyz_vx_tf > aabb_int_test_rev[..., None, 0,:], xyz_vx_tf < aabb_int_test_rev[..., None, 1,:]), axis=-1)
    
    in_pnts = jnp.logical_and(valid_vx[...,0], in_pnts)

    segment_id = jnp.logical_not(in_pnts).astype(jnp.int32)
    vx_data = jnp.broadcast_to(vx_data, jnp.broadcast_shapes(segment_id[...,None].shape, vx_data.shape))
    origin_shape = vx_data.shape
    vx_data = vx_data.reshape((-1,) + vx_data.shape[-2:])
    segment_id = segment_id.reshape((-1,) + segment_id.shape[-1:])
    positive_sum = jax.vmap(jax.ops.segment_sum, (0,0,None))(vx_data, segment_id, 1)
    positive_sum = positive_sum.reshape(origin_shape[:-2] + positive_sum.shape[-1:])
    positive_sum = positive_sum / (1e-6+jnp.sum(in_pnts.astype(jnp.float32), axis=-1, keepdims=True))
    
    return positive_sum


def vx_plane_intersection_idx(pos, quat, obj_vx, jkey, ns=10):
    valid_vx, xyz_vx, aabb_scaled, aabb_obj = obj_vx
    nv = valid_vx.shape[-2]
    if nv > 100:
        nv = jnp.round(nv**(1/3))
    
    aabb_ext = aabb_scaled[...,3:] - aabb_scaled[...,:3]
    xyz_vx_tf = tutil.pq_action(pos[...,None,:], quat[...,None,:], xyz_vx)
    in_pnts = xyz_vx_tf[...,2] <= 2.0*aabb_ext[...,None,0]/nv
    in_pnts = jnp.logical_and(valid_vx[...,0], in_pnts)
    random_idx = jax.random.permutation(jkey, in_pnts.shape[-1])
    where_idx = jax.vmap(lambda x : jnp.where(x, size=ns, fill_value=-1))(in_pnts.reshape((-1,) + in_pnts.shape[-1:])[:,random_idx])[0]
    where_idx = where_idx.reshape(in_pnts.shape[:-1]+(ns,))
    origin_idx = jnp.concatenate([random_idx, jnp.array([-1])], axis=0)[where_idx]

    return origin_idx


def vx_plane_intersection_mean(vx_data, pos, quat, obj_vx):
    valid_vx, xyz_vx, aabb_scaled, aabb_obj = obj_vx
    nv = valid_vx.shape[-2]
    if nv > 100:
        nv = jnp.round(nv**(1/3))
    
    aabb_ext = aabb_scaled[...,3:] - aabb_scaled[...,:3]
    xyz_vx_tf = tutil.pq_action(pos[...,None,:], quat[...,None,:], xyz_vx)
    in_pnts = xyz_vx_tf[...,2] <= 2.0*aabb_ext[...,None,0]/nv
    in_pnts = jnp.logical_and(valid_vx[...,0], in_pnts)

    segment_id = jnp.logical_not(in_pnts).astype(jnp.int32)
    vx_data = jnp.broadcast_to(vx_data, jnp.broadcast_shapes(segment_id[...,None].shape, vx_data.shape))
    origin_shape = vx_data.shape
    vx_data = vx_data.reshape((-1,) + vx_data.shape[-2:])
    segment_id = segment_id.reshape((-1,) + segment_id.shape[-1:])
    positive_sum = jax.vmap(jax.ops.segment_sum, (0,0,None))(vx_data, segment_id, 1)
    positive_sum = positive_sum.reshape(origin_shape[:-2] + positive_sum.shape[-1:])
    positive_sum = positive_sum / (1e-6+jnp.sum(in_pnts.astype(jnp.float32), axis=-1, keepdims=True))
    
    return positive_sum