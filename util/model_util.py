import jax.numpy as jnp
import einops
import flax.linen as nn
import numpy as np
import jax
import open3d as o3d

import util.transform_util as tutil
import util.voxel_util as vutil

IDX_DIR = jnp.array([[0,0,0,0,0,0],
                    [1,0,0,0,0,0],[-1,0,0,0,0,0],
                    [0,1,0,0,0,0],[0,-1,0,0,0,0],
                    [0,0,1,0,0,0],[0,0,-1,0,0,0],
                    [0,0,0,1,0,0],[0,0,0,-1,0,0],
                    [0,0,0,0,1,0],[0,0,0,0,-1,0],
                    [0,0,0,0,0,1],[0,0,0,0,0,-1]]).astype(jnp.float32)


class Model(nn.Module):
    base_dim:int=128
    num_layers:int=3

    @nn.compact
    def __call__(self, x, epsilon, z):
        z = einops.rearrange(z, '... i j -> ... (i j)')
        
        pos, quat = x
        pos1, pos2 = pos[...,0,:], pos[...,1,:]
        quat1, quat2 = quat[...,0,:], quat[...,1,:]
        pos, quat = tutil.pq_multi(*tutil.pq_inv(pos1, quat1), pos2, quat2)
        alpha = self.param('pos_log_scale', nn.initializers.zeros, (), jnp.float32)
        alpha_eps = self.param('epsilon_log_scale', nn.initializers.zeros, (2,), jnp.float32)

        R_flat = einops.rearrange(tutil.q2R(quat), '... i j -> ... (i j)')
        x = jnp.concatenate([pos * jnp.exp(alpha), R_flat, jnp.exp(alpha_eps)*epsilon, z], axis=-1)
        skip = x
        for i in range(self.num_layers):
            x = nn.Dense(self.base_dim)(x)
            x = nn.relu(x)
            if (self.num_layers+1)//2 == i:
                x = jnp.concatenate([x, skip], axis=-1)

        x = nn.Dense(13)(x)
        x = nn.tanh(x)
        return x


class Model1(nn.Module):
    base_dim:int=128
    num_layers:int=3

    @nn.compact
    def __call__(self, x, z):
        z = einops.rearrange(z, '... i j -> ... (i j)')
        
        pos, quat = x
        pos1, pos2 = pos[...,0,:], pos[...,1,:]
        quat1, quat2 = quat[...,0,:], quat[...,1,:]
        pos, quat = tutil.pq_multi(*tutil.pq_inv(pos1, quat1), pos2, quat2)
        alpha = self.param('pos_log_scale', nn.initializers.zeros, (), jnp.float32)

        R_flat = einops.rearrange(tutil.q2R(quat), '... i j -> ... (i j)')
        x = jnp.concatenate([pos * jnp.exp(alpha), R_flat, z], axis=-1)
        skip = x
        for i in range(self.num_layers):
            x = nn.Dense(self.base_dim)(x)
            x = nn.relu(x)
            if (self.num_layers+1)//2 == i:
                x = jnp.concatenate([x, skip], axis=-1)

        x = nn.Dense(1)(x)
        x = nn.tanh(x)
        # x = jnp.squeeze(x, axis=-1)
        return x

def voxel_vector_max(xyz, vectors_per_xyz, voxel_size):

    npnt = xyz.shape[-2]

    aabb = jnp.concatenate([jnp.min(xyz, axis=-2, keepdims=True)-0.005, jnp.max(xyz, axis=-2, keepdims=True)+0.005], axis=-1)
    max_len = jnp.max(aabb[...,3:] - aabb[...,:3], axis=-1)
    aabb_scaled = jnp.stack([-max_len,-max_len,-max_len,max_len,max_len,max_len], axis=-1) * 0.5 + einops.repeat(0.5*(aabb[...,:3] + aabb[...,3:]), '... i -> ... (i j)', j=2)
    
    xyz_scaled = (xyz - aabb_scaled[...,:3])/(aabb_scaled[...,3:]-aabb_scaled[...,:3]) * voxel_size
    xyz_idx = jnp.floor(xyz_scaled)
    xyz_normalized = xyz_scaled - xyz_idx
    xyz_idx_flat = voxel_size**2*xyz_idx[...,0] + voxel_size*xyz_idx[...,1] + xyz_idx[...,2]
    
    origin_outer_shape = xyz_idx_flat.shape[:-1]
    x, xyz_idx_flat = [el.reshape((-1,)+el.shape[len(origin_outer_shape):]) for el in (vectors_per_xyz, xyz_idx_flat)]
    x_vx = jax.vmap(jax.ops.segment_max, (0,0,None))(x, xyz_idx_flat.astype(jnp.int32), voxel_size**3)
    valid_vx = jnp.all(x_vx!=-jnp.inf, axis=-1, keepdims=True).astype(jnp.float32)
    x_vx = jnp.where(x_vx==-jnp.inf, 0, x_vx)

    x_vx, valid_vx = [el.reshape(origin_outer_shape + el.shape[-2:]) for el in (x_vx, valid_vx)]

    # valid_vx, x_vx = [einops.rearrange(el, '... (p q k) j -> ... p q k j', p=voxel_size, q=voxel_size, k=voxel_size) for el in (valid_vx, x_vx)]
    xyz_vx = jnp.stack(jnp.meshgrid(jnp.arange(voxel_size), jnp.arange(voxel_size), jnp.arange(voxel_size), indexing='ij'), axis=-1)
    xyz_vx = xyz_vx.reshape(-1,3)
    xyz_vx = (xyz_vx + 0.5)/voxel_size * (aabb_scaled[...,3:] - aabb_scaled[...,:3]) + aabb_scaled[...,:3]

    return x_vx, xyz_normalized, (valid_vx, xyz_vx, jnp.squeeze(aabb_scaled, axis=-2), jnp.squeeze(aabb, axis=-2))

class PCDEncoder(nn.Module):
    base_dim:int=64
    num_layers:int=3
    voxel_size:int=7
    latent_type1:str='discrete'
    latent_type2:str='vv'
    feature_dim:int=32
    voxel_factor:int=1

    @nn.compact
    def __call__(self, xyz, normal):
        
        npnt = xyz.shape[-2]

        aabb = jnp.concatenate([jnp.min(xyz, axis=-2, keepdims=True)-0.01, jnp.max(xyz, axis=-2, keepdims=True)+0.01], axis=-1)
        aabb_origin = aabb
        max_len = jnp.max(aabb[...,3:] - aabb[...,:3], axis=-1)
        aabb_scaled = jnp.stack([-max_len,-max_len,-max_len,max_len,max_len,max_len], axis=-1) * 0.5 + einops.repeat(0.5*(aabb[...,:3] + aabb[...,3:]), '... i -> ... (i j)', j=2)
        aabb = aabb_scaled
        
        first_voxel_size = self.voxel_size * self.voxel_factor
        xyz_scaled = (xyz - aabb[...,:3])/(aabb[...,3:]-aabb[...,:3]) * first_voxel_size
        xyz_idx = jnp.floor(xyz_scaled)
        xyz_normalized = xyz_scaled - xyz_idx
        xyz_idx_flat = first_voxel_size**2*xyz_idx[...,0] + first_voxel_size*xyz_idx[...,1] + xyz_idx[...,2]
        aabb_log_scale = self.param('aabb_log_scale', nn.initializers.zeros, (), jnp.float32)

        ## feature extraction
        # x = jnp.concatenate([xyz_normalized * 2 - 1, normal, einops.repeat(jnp.exp(aabb_log_scale)*aabb, '... i j -> ... (i k) j', k=npnt)], axis=-1)
        x = jnp.concatenate([xyz, xyz_normalized * 2 - 1, normal], axis=-1)

        if self.voxel_factor == 1:
            for i in range(self.num_layers):
                x = nn.Dense(self.base_dim)(x)
                x = nn.relu(x)
                if i == 0:
                    skip = x
            x += skip
        else:
            for i in range(self.num_layers):
                x = nn.Dense(self.base_dim//2)(x)
                x = nn.relu(x)
                if i == 0:
                    skip = x
            x += skip
        
        origin_outer_shape = xyz_idx_flat.shape[:-1]
        x, xyz_idx_flat = [el.reshape((-1,)+el.shape[len(origin_outer_shape):]) for el in (x, xyz_idx_flat)]
        x_vx = jax.vmap(jax.ops.segment_max, (0,0,None))(x, xyz_idx_flat.astype(jnp.int32), first_voxel_size**3)
        valid_vx = jnp.all(x_vx!=-jnp.inf, axis=-1, keepdims=True).astype(jnp.float32)
        x_vx = jnp.where(x_vx==-jnp.inf, 0, x_vx)
        x_vx = jnp.concatenate([valid_vx, x_vx], axis=-1)
        x_vx = einops.rearrange(x_vx, '... (i p q) j -> ... i p q j', i=first_voxel_size, p=first_voxel_size, q=first_voxel_size)


        xyz_vx = jnp.stack(jnp.meshgrid(jnp.arange(first_voxel_size), jnp.arange(first_voxel_size), jnp.arange(first_voxel_size), indexing='ij'), axis=-1)
        aabb_scaled_float = aabb_scaled.reshape((-1,) + aabb_scaled.shape[-2:])
        aabb_scaled_float = aabb_scaled_float[...,None,None,:]
        xyz_vx = (xyz_vx + 0.5) / first_voxel_size * (aabb_scaled_float[...,3:] - aabb_scaled_float[...,:3]) + aabb_scaled_float[...,:3]
        
        abbb_vx = aabb_origin.reshape((-1,) + aabb_origin.shape[-1:])
        abbb_vx = einops.repeat(jnp.exp(aabb_log_scale)*abbb_vx, '... i j -> ... i p q t j', p=first_voxel_size, q=first_voxel_size, t=first_voxel_size)
        x_vx = jnp.concatenate([x_vx, xyz_vx, abbb_vx], axis=-1)

        # # ## 3d convolusion
        if self.voxel_factor > 1:
            valid_vx = einops.rearrange(valid_vx, '... (i ii p pp q qq) j -> ... i p q ii pp qq j', i=self.voxel_size, p=self.voxel_size, q=self.voxel_size,
                            ii=self.voxel_factor, pp=self.voxel_factor, qq=self.voxel_factor)
            valid_vx = jnp.max(valid_vx, axis=(-2,-3,-4))
            valid_vx = valid_vx.reshape(valid_vx.shape[:-4] + (-1,1))

            for _ in range(1):
                x_vx = nn.Conv(self.base_dim//2, kernel_size=(3,3,3), padding='SAME')(x_vx)
                x_vx = nn.relu(x_vx)
            x_vx = nn.Conv(self.base_dim//2, kernel_size=(3,3,3), strides=(2,2,2), padding='SAME')(x_vx)
            x_vx = nn.relu(x_vx)

        ltype = []
        skips = []
        for l in range(self.num_layers):
            if x_vx.shape[-2] >= 5:
                x_vx = nn.Conv(self.base_dim, kernel_size=(3,3,3), padding='VALID')(x_vx)
                ltype.append(0)
            else:
                x_vx = nn.Conv(self.base_dim, kernel_size=(3,3,3), padding='SAME')(x_vx)
                ltype.append(1)
            x_vx = nn.relu(x_vx)
            x_vx = nn.LayerNorm()(x_vx)
            if l != self.num_layers-1:
                skips.append(x_vx)
        gloval_fts = jnp.mean(x_vx, axis=(-2,-3,-4), keepdims=True)
        if self.latent_type2=='gg':
            gloval_fts = gloval_fts.reshape(origin_outer_shape + gloval_fts.shape[-1:])
            return (None, None, gloval_fts), None, None

        skips = skips[::-1]
        for l, lt in enumerate(ltype[::-1]):
            if lt == 0:
                x_vx = nn.ConvTranspose(self.base_dim, kernel_size=(3,3,3), padding='VALID')(x_vx)
            else:
                x_vx = nn.ConvTranspose(self.base_dim, kernel_size=(3,3,3), padding='SAME')(x_vx)
            x_vx = nn.relu(x_vx)
            x_vx = nn.LayerNorm()(x_vx)
            if l != self.num_layers-1:
                x_vx = jnp.concatenate([x_vx, skips[l]], axis=-1)

        x_vx = jnp.concatenate([x_vx, einops.repeat(gloval_fts, '... i j k p -> ... (i v1) (j v2) (k v3) p', v1=x_vx.shape[-4], v2=x_vx.shape[-3], v3=x_vx.shape[-2])], axis=-1)
        x_vx = nn.Dense(self.base_dim)(x_vx)
        x_vx = nn.relu(x_vx)
        x_vx = nn.LayerNorm()(x_vx)

        valid_vx = valid_vx.reshape(origin_outer_shape+ valid_vx.shape[-2:])
        gloval_fts = gloval_fts.reshape(origin_outer_shape + gloval_fts.shape[-1:])
        x_vx = x_vx.reshape(origin_outer_shape+x_vx.shape[-4:])
        
        x_vx = nn.Dense(self.feature_dim)(x_vx)
        return (valid_vx, None, gloval_fts), x_vx, (jnp.squeeze(aabb, axis=-2), jnp.squeeze(aabb_origin, axis=-2))



    
class DecModel(nn.Module):
    base_dim:int=32
    num_layers:int=3
    ns:int=10
    latent_type1:str='discrete'
    latent_type2:str='vv'
    cls_type:int=1
    local_type:int=0

    @nn.compact
    def __call__(self, x, emb_vx, aabb=None, jkey=None, xz=None):
        output_dim = (13 if self.cls_type==1 else 1)
        valid_vx, z_vx_idx, global_fts = emb_vx
        if valid_vx is not None:
            nv = int(np.round(valid_vx.shape[-2] ** (1/3)))
        if aabb is not None:
            aabb_scaled, aabb_origin = aabb

        pos, quat = x
        pos1, pos2 = pos[...,0,:], pos[...,1,:]
        quat1, quat2 = quat[...,0,:], quat[...,1,:]
        relpos, relquat = tutil.pq_multi(*tutil.pq_inv(pos1, quat1), pos2, quat2)
        alpha = self.param('pos_log_scale', nn.initializers.zeros, (), jnp.float32)
        relR_flat = einops.rearrange(tutil.q2R(relquat), '... i j -> ... (i j)')

        if self.latent_type2=='gg':
            global_fts_flat = einops.rearrange(global_fts, '... i j -> ... (i j)')
            global_fts_flat = jnp.broadcast_to(global_fts_flat, jnp.broadcast_shapes(relpos[...,0:1].shape, global_fts_flat.shape))
            x = jnp.concatenate([relpos * jnp.exp(alpha), relR_flat, global_fts_flat], axis=-1)
            skip = x
            for i in range(self.num_layers):
                x = nn.Dense(self.base_dim)(x)
                x = nn.relu(x)
                if (self.num_layers-1)//2 == i:
                    x = jnp.concatenate([x, skip], axis=-1)
            x = nn.Dense(output_dim)(x)
            x = nn.tanh(x)
            return x

        relpos_pd = jnp.stack([jnp.zeros_like(relpos), relpos], axis=-2)
        relquat_pd = jnp.stack([tutil.qExp(jnp.zeros_like(relpos)), relquat], axis=-2)

        xyz_vx = jnp.stack(jnp.meshgrid(jnp.arange(nv), jnp.arange(nv), jnp.arange(nv), indexing='ij'), axis=-1)
        xyz_vx = einops.rearrange(xyz_vx, '... i j k p -> ... (i j k) p')
        xyz_vx = (xyz_vx + 0.5) / nv * (aabb_scaled[...,None,3:] - aabb_scaled[...,None,:3]) + aabb_scaled[...,None,:3]
        xyz_vx_tf = tutil.pq_action(relpos_pd[...,None,:], relquat_pd[...,None,:], xyz_vx)

        obj_vx = (valid_vx, xyz_vx, aabb_scaled, aabb_origin)

        if self.local_type==0:
            widx = vutil.vx_intersection_idx(*x, obj_vx, jkey, self.ns)

            z = jnp.take_along_axis(einops.rearrange(xz, '... i j k p -> ... (i j k) p'), einops.rearrange(widx, '... i -> ... i 1'), axis=-2)

            pick_xyz_vx_tf = jnp.take_along_axis(xyz_vx_tf, widx[...,None], axis=-2)
            pose_fts = jnp.concatenate([relpos * jnp.exp(alpha), relR_flat], axis=-1)
            pose_fts = einops.repeat(pose_fts, '... i -> ... p k i', p=1, k=z.shape[-2])
            pose_fts = jnp.concatenate([jnp.zeros_like(pose_fts), pose_fts], axis=-3)
            z = jnp.concatenate([pick_xyz_vx_tf, z, pose_fts], axis=-1)
            
            ## simple concat features
            for i in range(self.num_layers//2):
                z = nn.Dense(self.base_dim//2)(z)
                z = nn.relu(z)
                z = nn.LayerNorm()(z)
                if i==0:
                    skip = z
            z+=skip
            z_pair = jnp.sum(z, axis=-2)
            z = einops.rearrange(z_pair, '... i j -> ... (i j)')

            for i in range(self.num_layers - self.num_layers//2):
                z = nn.Dense(self.base_dim)(z)
                z = nn.relu(z)
                z = nn.LayerNorm()(z)
                if i==0:
                    skip = z
            z += skip

            z = nn.Dense(output_dim)(z)
            z = nn.tanh(z)
            return z

        elif self.local_type == 1:
            xz = einops.rearrange(xz, '... i j k p -> ... (i j k) p')
            z = vutil.vx_intersection_mean(xz, pos, quat, obj_vx, jkey)
            z = jnp.reshape(z, z.shape[:-2] + (-1,))
            pose_fts = jnp.concatenate([relpos * jnp.exp(alpha), relR_flat], axis=-1)
            z = jnp.concatenate([z, pose_fts], axis=-1)

            for i in range(self.num_layers):
                z = nn.Dense(self.base_dim)(z)
                z = nn.relu(z)
                z = nn.LayerNorm()(z)
                if i==0:
                    skip = z
            z += skip

            z = nn.Dense(output_dim)(z)
            z = nn.tanh(z)

            return z

class RenderModel(nn.Module):
    num_layers:int=4
    base_dim:int=128
    cls_type:int=0
    latent_type1:str='continuous'
    latent_type2:str='vv'

    @nn.compact
    def __call__(self, x, emb_vx, aabb, jkey, xz, len):
        valid_vx, z_vx_idx, global_fts = emb_vx
        if valid_vx is not None:
            nv = int(np.round(valid_vx.shape[-2] ** (1/3)))
        aabb_scaled, aabb_origin = aabb

        pos, quat = x
        pos1, pos2 = pos[...,0,:], pos[...,1,:]
        quat1, quat2 = quat[...,0,:], quat[...,1,:]
        relpos, relquat = tutil.pq_multi(*tutil.pq_inv(pos1, quat1), pos2, quat2)
        raydir = tutil.q2R(relquat)[...,:,2]

        
        # dir = tutil.normalize(dir)
        num_out = 5
        num_feature_pnts = 7
        scattered = jnp.linspace(-0.5, 0.5, num=num_feature_pnts, endpoint=True)
        pnts = relpos[...,None,:] + raydir[...,None,:] * scattered[...,None] * len[...,None,None] * (0.5 * num_out + 0.5)
        # ray_center_pnts = relpos[...,None,:] + raydir[...,None,:] * jnp.linspace(-0.5, 0.5, num=num_out, endpoint=True) * len[...,None,None] * (0.5 * num_out - 0.5)

        idx_continue = (pnts - aabb_scaled[...,None,:3]) / (aabb_scaled[...,None,3:]-aabb_scaled[...,None,:3]) * nv
        vx_idx = jnp.floor(idx_continue)
        vx_idx = jnp.clip(vx_idx, 0, nv-1)
        vx_idx_flat = nv**2*vx_idx[...,0] + nv*vx_idx[...,1] + vx_idx[...,2]
        pos_normalized = idx_continue - vx_idx
        pos_normalized = pos_normalized*2.0-1.0

        xz = xz.reshape((xz.shape[:-4] + (-1,xz.shape[-1])))
        z = jnp.take_along_axis(xz, vx_idx_flat[...,None].astype(jnp.int32), axis=-2)

        z = jnp.concatenate([z, pos_normalized, pnts], axis=-1)
        for i in range(self.num_layers//2):
            z = nn.Dense(self.base_dim)(z)
            z = nn.relu(z)
            if i== (self.num_layers//2-1)//2:
                    skip = z
        z += skip
        # z = jnp.concatenate([z.reshape(z.shape[:-2] + (-1,)), pos_normalized.reshape(pos_normalized.shape[:-2] + (-1,))], axis=-1)
        z = z.reshape((z.shape[:-2]+(-1,)))
        for i in range(self.num_layers-self.num_layers//2):
            z = nn.Dense(self.base_dim)(z)
            z = nn.relu(z)
            if i== (self.num_layers-self.num_layers//2-1)//2:
                skip = z
        z += skip

        # z = nn.Dense(1)(z)
        z = nn.Dense(num_out)(z)
        z = nn.tanh(z)

        return z


class OCCModel(nn.Module):
    num_layers:int=4
    base_dim:int=128
    latent_type1:str='discrete'
    latent_type2:str='vv'

    @nn.compact
    def __call__(self, x, emb_vx, aabb, jkey, xz=None):
        valid_vx, z_vx_idx, global_fts = emb_vx
        if valid_vx is not None:
            nv = int(np.round(valid_vx.shape[-2] ** (1/3)))
        if aabb is not None:
            aabb_scaled, aabb_origin = aabb

        # if z_vx_idx is not None:
        if self.latent_type2=='vv':
            pnts = x
            idx_continue = (pnts - aabb_scaled[...,:3]) / (aabb_scaled[...,3:]-aabb_scaled[...,:3]) * nv
            vx_idx = jnp.floor(idx_continue)
            vx_idx = jnp.clip(vx_idx, 0, nv-1)
            vx_idx_flat = nv**2*vx_idx[...,0] + nv*vx_idx[...,1] + vx_idx[...,2]
            pos_normalized = idx_continue - vx_idx
            pos_normalized = pos_normalized*2.0-1.0

            xz = xz.reshape((xz.shape[:-4] + (-1, xz.shape[-1])))
            xz = jnp.take_along_axis(xz, vx_idx_flat[...,None, None].astype(jnp.int32), axis=-2)
            z = jnp.squeeze(xz, axis=-2)
            z = jnp.concatenate([z, pos_normalized, pnts], axis=-1)
        else:
            global_fts = jnp.broadcast_to(global_fts, jnp.broadcast_shapes(x[...,0:1].shape, global_fts.shape))
            z = jnp.concatenate([global_fts, x], axis=-1)
        for i in range(self.num_layers):
            z = nn.Dense(self.base_dim)(z)
            z = nn.relu(z)
            if i== (self.num_layers-1)//2:
                    skip = z
        z += skip

        z = nn.Dense(1)(z)
        z = nn.tanh(z)

        return z



class PlaneModel(nn.Module):
    base_dim:int=32
    num_layers:int=3
    cls_type:int=1
    ns:int=8
    latent_type2:str='vv'
    local_type:int=0

    @nn.compact
    def __call__(self, x, emb_vx, aabb=None, jkey=None, xz=None):
        output_dim = (13 if self.cls_type==1 else 1)
        valid_vx, z_vx_idx, global_fts = emb_vx
        if valid_vx is not None:
            nv = int(np.round(valid_vx.shape[-2] ** (1/3)))
        if aabb is not None:
            aabb_scaled, aabb_origin = aabb

        pos, quat = x
        alpha = self.param('pos_log_scale', nn.initializers.zeros, (), jnp.float32)
        R_flat = einops.rearrange(tutil.q2R(quat), '... i j -> ... (i j)')

        if self.latent_type2 == 'vv':
            xyz_vx = jnp.stack(jnp.meshgrid(jnp.arange(nv), jnp.arange(nv), jnp.arange(nv), indexing='ij'), axis=-1)
            xyz_vx = einops.rearrange(xyz_vx, '... i j k p -> ... (i j k) p')
            xyz_vx = (xyz_vx + 0.5) / nv * (aabb_scaled[...,None,3:] - aabb_scaled[...,None,:3]) + aabb_scaled[...,None,:3]
            xyz_vx_tf = tutil.pq_action(pos[...,None,:], quat[...,None,:], xyz_vx)

            obj_vx = (valid_vx, xyz_vx, aabb_scaled, aabb_origin)
            if self.local_type==0:
                widx = vutil.vx_plane_intersection_idx(*x, obj_vx, jkey, self.ns)

                z = jnp.take_along_axis(einops.rearrange(xz, '... i j k p -> ... (i j k) p'), einops.rearrange(widx, '... i -> ... i 1'), axis=-2)

                pick_xyz_vx_tf = jnp.take_along_axis(xyz_vx_tf, widx[...,None], axis=-2)
                pose_fts = jnp.concatenate([pos * jnp.exp(alpha), R_flat], axis=-1)
                pose_fts = einops.repeat(pose_fts, '... i -> ... k i', k=z.shape[-2])
                # pose_fts = jnp.concatenate([jnp.zeros_like(pose_fts), pose_fts], axis=-3)
                z = jnp.concatenate([pick_xyz_vx_tf, z, pose_fts], axis=-1)
                
                ## simple concat features
                for i in range(self.num_layers//2):
                    z = nn.Dense(self.base_dim//2)(z)
                    z = nn.relu(z)
                    z = nn.LayerNorm()(z)
                    if i==0:
                        skip = z
                z+=skip
                z = jnp.sum(z, axis=-2)

                for i in range(self.num_layers - self.num_layers//2):
                    z = nn.Dense(self.base_dim)(z)
                    z = nn.relu(z)
                    z = nn.LayerNorm()(z)
                    if i==0:
                        skip = z
                z += skip
            elif self.local_type==1:
                xz = einops.rearrange(xz, '... i j k p -> ... (i j k) p')
                z = vutil.vx_plane_intersection_mean(xz, *x, obj_vx)
                pose_fts = jnp.concatenate([pos * jnp.exp(alpha), R_flat], axis=-1)
                z = jnp.concatenate([z, pose_fts], axis=-1)

                for i in range(self.num_layers):
                    z = nn.Dense(self.base_dim)(z)
                    z = nn.relu(z)
                    z = nn.LayerNorm()(z)
                    if i==0:
                        skip = z
                z += skip
        else:
            global_fts = jnp.broadcast_to(global_fts, jnp.broadcast_shapes(pos[...,0:1].shape, global_fts.shape))
            z = jnp.concatenate([global_fts, pos * jnp.exp(alpha), R_flat], axis=-1)
            for i in range(self.num_layers):
                z = nn.Dense(self.base_dim)(z)
                z = nn.relu(z)
                if i==0:
                    skip = z
            z += skip

        z = nn.Dense(output_dim)(z)
        z = nn.tanh(z)

        return z



def voxel_value_hashing(nv, pnts, aabb):
    idx_continue = (pnts - aabb[...,:3]) / (aabb[...,3:]-aabb[...,:3]) * nv - 0.5
    vx_idx = jnp.floor(idx_continue)
    vx_idx = vx_idx[...,None,:] + jnp.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])
    vx_idx = jnp.clip(vx_idx, 0, nv-1)
    dist = jnp.abs(idx_continue[...,None,:] - vx_idx)
    weights = jnp.prod(dist, axis=-1)[...,(7,6,5,4,3,2,1,0),None]
    return vx_idx.astype(jnp.int32), weights/(jnp.sum(weights, axis=-2, keepdims=True) + 1e-6)
        

def center_to_sphere_pnts(cpos, cquat):
    return tutil.pq_multi(IDX_DIR[...,:3], tutil.qExp(IDX_DIR[...,3:]), cpos[...,None,:], cquat[...,None,:])

import os
import pickle

def broadcast_var(target_var, var):
    return  jnp.broadcast_to(var, jnp.broadcast_shapes(target_var.shape, var.shape))

def model_import(save_dir, pcd_out=False):
    with open(os.path.join(save_dir, 'mesh_dir_list.pkl'), 'rb') as f:
        mesh_dir_list = pickle.load(f)
    with open(os.path.join(save_dir, 'params.pkl'), 'rb') as f:
        raw_loaded = pickle.load(f)
    with open(os.path.join(save_dir, 'primitive_params.pkl'), 'rb') as f:
        primitive_params = pickle.load(f)

    config = raw_loaded['config']
    params = raw_loaded['params']
    
    if pcd_out:
        with open(os.path.join('dataset', 'pcd_list1500.pkl'), 'rb') as f:
            raw_pcd_list = pickle.load(f)
        mesh_dir_list = np.array(list(raw_pcd_list.keys()))
        pcd_list = jnp.array([raw_pcd_list[rp][0] for rp in raw_pcd_list])
        normal_list = jnp.array([raw_pcd_list[rp][1] for rp in raw_pcd_list])

    enc_key_list = ['enc_num_layers', 'enc_base_dim', 'voxel_size', 'latent_type1', 'latent_type2', 'feature_dim']
    dec_key_list = ['dec_num_layers', 'dec_base_dim', 'ns', 'latent_type1', 'latent_type2', 'cls_type', 'local_type']
    pln_key_list = ['pln_num_layers', 'pln_base_dim', 'cls_type', 'latent_type2', 'local_type']
    occ_key_list = ['occ_num_layers', 'occ_base_dim', 'latent_type1', 'latent_type2']
    enc_model = PCDEncoder(**{(k if k[:3]!='enc' else k[4:]):config[k] for k in config if k in enc_key_list})
    dec_model = DecModel(**{(k if k[:3]!='dec' else k[4:]):config[k] for k in config if k in dec_key_list})
    pln_model = PlaneModel(**{(k if k[:3]!='pln' else k[4:]):config[k] for k in config if k in pln_key_list})
    occ_model = OCCModel(**{(k if k[:3]!='occ' else k[4:]):config[k] for k in config if k in occ_key_list})

    if pcd_out:
        def enc_func(z_idx, pcd_list=pcd_list, normal_list=normal_list):
            xyz = pcd_list[z_idx]
            normal = normal_list[z_idx]
            return enc_model.apply(params[0], xyz, normal)
    else:
        def enc_func(xyz, normal):
            return enc_model.apply(params[0], xyz, normal)

    def dec_func(x, emb, aabb, jkey, xz=None):
        if config['latent_type1'] == 'discrete':
            return dec_model.apply(params[1], x, emb, aabb, jkey)
        else:
            return dec_model.apply(params[1], x, emb, aabb, jkey, xz)

    def occ_func(x, emb, aabb, jkey, xz):
        return occ_model.apply(params[3], x, emb, aabb, jkey, xz)

    def pln_func(x, emb, aabb, jkey, xz):
        return pln_model.apply(params[4], x, emb, aabb, jkey, xz)

    if pcd_out:
        return config, mesh_dir_list, primitive_params, dict(enc=enc_func, dec=dec_func, pln=pln_func, occ=occ_func), (pcd_list, normal_list)
    else:
        return config, dict(enc=enc_func, dec=dec_func, pln=pln_func, occ=occ_func)