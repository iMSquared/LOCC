from typing import NamedTuple
import jax.numpy as jnp
import einops
import numpy as np
import jax

import util.transform_util as tutil


class SimParams(NamedTuple):
    friction_coef_pln:float
    friction_coef_obj:float
    rolling_friction_coef:float
    inertia:jnp.ndarray
    drag_v:float
    drag_w:float
    elasticity:float=0.0
    dt:float=0.010
    mass:float=1
    baumgarte_erp_pln:float=290
    baumgarte_erp_obj:float=290


# %%
# simulation
def culling(states, plane=False):
    nb, no = nb_no_from_states(states)
    indices = jnp.triu_indices(no, k=1)
    if plane:
        idx1 = jnp.concatenate([indices[0], jnp.arange(no)], axis=0)
        idx2 = jnp.concatenate([indices[1], no+jnp.arange(no)], axis=0)
    else:
        idx1 = indices[0]
        idx2 = indices[1]

    return einops.repeat(idx1, 'i -> b i', b=nb), einops.repeat(idx2, 'i -> b i', b=nb)

def nb_no_from_states(states):
    return states[0].shape[0], states[0].shape[1]

def global_to_cull_idx(var, idx):
    for _ in range(len(var.shape) - len(idx.shape)):
        idx = idx[...,None]
    return jnp.take_along_axis(var, idx, axis=1)

def get_const(states12, jkey, funcs):

    def jaco_func(x2, x1, obj_variables, jkey):
        x = jax.tree_util.tree_map(lambda *x : jnp.stack(x, axis=-2), x1, x2)
        return funcs['dec'](x, obj_variables[0], obj_variables[1], jkey, obj_variables[2])

    jaco_vmap = jax.vmap(jax.jacobian(jaco_func), (0,0,0,None))

    origin_outer = states12[0].shape[:-2]
    states12 = jax.tree_util.tree_map(lambda x: einops.rearrange(x, 'i j ... -> (i j) ... '), states12)
    x1 = jax.tree_util.tree_map(lambda x : x[...,0,:], states12[:2])
    x2 = jax.tree_util.tree_map(lambda x : x[...,1,:], states12[:2])
    jacos = jaco_vmap(x2, x1, states12[4], jkey)

    qtau = tutil.qvee(tutil.qmulti(tutil.qinv(x2[1][...,None,:]), jacos[1]))
    w = tutil.qaction(x2[1][...,None,:], qtau)

    w_scale = 1.0
    const = tutil.normalize(jnp.concatenate((jacos[0], w*w_scale), axis=-1))
    const = einops.rearrange(const, '(i j) ... -> i j ...', i=origin_outer[0], j=origin_outer[1])

    return const


def get_const_pln(states, jkey, funcs):

    def pln_jaco_func(x, obj_variables, jkey):
        return funcs['pln'](x, obj_variables[0], obj_variables[1], jkey, obj_variables[2])

    pln_jaco_vmap = jax.vmap(jax.jacobian(pln_jaco_func), (0,0,None))

    origin_outer = states[0].shape[:-1]
    states = jax.tree_util.tree_map(lambda x: einops.rearrange(x, 'i j ... -> (i j) ... '), states)
    jacos = pln_jaco_vmap(states[:2], states[4], jkey)

    qtau = tutil.qvee(tutil.qmulti(tutil.qinv(states[1][...,None,:]), jacos[1]))
    w = tutil.qaction(states[1][...,None,:], qtau)
    
    w_scale = 1.0
    const = tutil.normalize(jnp.concatenate((jacos[0], w*w_scale), axis=-1))
    const = einops.rearrange(const, '(i j) ... -> i j ...', i=origin_outer[0], j=origin_outer[1])

    return const


def cal_response(rel_v, rel_w, rel_cps, cns, ic_values, const, sim_params:SimParams, rel_pos=None, type='pln'):
    '''
    rel_v of objects : (... 3)
    rel_w of objects : (... 3)
    rel_cps : (... NC 3)
    cns : (... NC 3)
    ic_values : (... NC)
    const : (... NC 6)
    rel_pos : (... 3)
    '''
    # cp vels
    vel_cps = rel_v[...,None,:] + jnp.cross(rel_w[...,None,:], rel_cps)
    vel_cps_n_mag = jnp.einsum('...i,...i', vel_cps, cns)[...,None]
    vel_cps_n = vel_cps_n_mag * cns
    vel_cps_t = vel_cps - vel_cps_n
    vel_cps_t_mag = jnp.linalg.norm(vel_cps_t, axis=-1, keepdims=True)
    vel_cps_t_normalized = tutil.normalize(vel_cps_t)
    
    # contact force
    berp = sim_params.baumgarte_erp_obj if rel_pos is not None else sim_params.baumgarte_erp_pln
    baumgarte_vel = -berp * ic_values[...,None] / 1000.
    temp1 = jnp.einsum('...ki,...ji->...jk', sim_params.inertia, jnp.cross(rel_cps, cns))
    ang = jnp.einsum('...i,...i', cns, jnp.cross(temp1, rel_cps))[...,None]
    apply_veln = jnp.where(
        (ic_values[...,None] < 0) & (vel_cps_n_mag < 0), 1.,
        0.)
    impulse = (-1. * (1. + sim_params.elasticity) * vel_cps_n_mag * apply_veln / jnp.sum(apply_veln,-2,keepdims=True).clip(1e-10) + baumgarte_vel) / (
        (1. / sim_params.mass) + ang)
    impulse_n_vec = impulse * cns
    response_n = jnp.concatenate([impulse_n_vec, jnp.cross(rel_cps, impulse_n_vec)], axis=-1)

    # friction
    fiction = sim_params.friction_coef_pln if type=='pln' else sim_params.friction_coef_obj
    impulse_d = vel_cps_t_mag / ((1. / sim_params.mass) + ang)
    impulse_d = jnp.minimum(impulse_d, fiction * impulse)
    impulse_d_vec = -impulse_d * vel_cps_t_normalized
    response_d = jnp.concatenate([impulse_d_vec, jnp.cross(rel_cps, impulse_d_vec)], axis=-1)

    ## approximated rolling + torsion friction
    rel_w_mag = jnp.linalg.norm(rel_w[...,None,:], axis=-1, keepdims=True) 
    impulse_d_roll = rel_w_mag / ((1. / sim_params.mass) + ang)
    impulse_d_roll = jnp.minimum(impulse_d_roll, sim_params.rolling_friction_coef * impulse)
    impulse_d_roll_vec = -impulse_d_roll * tutil.normalize(rel_w[...,None,:])
    response_d_roll = jnp.concatenate([jnp.zeros_like(impulse_d_roll_vec), impulse_d_roll_vec], axis=-1)

    # apply collision if penetrating, approaching, and oriented correctly
    vel_const_n_mag = jnp.einsum('...i,...i', const, jnp.c_[rel_v[...,None,:], rel_w[...,None,:]])[...,None]
    apply_n = jnp.where(
        (ic_values[...,None] < 0) & (impulse > 0.), 1.,
        0.)
    # apply drag if moving laterally above threshold
    apply_d = apply_n * jnp.where(vel_cps_t_mag > 0., 1., 0.)
    apply_d_roll = apply_n * jnp.where(rel_w_mag > 0., 1., 0.)

    response = response_n * apply_n + response_d * apply_d + response_d_roll * apply_d_roll
    response = jnp.sum(response, axis=-2) 

    if rel_pos is not None:
        rel_cpsA = rel_cps + rel_pos[...,None,:]
        response_nA = jnp.concatenate([-impulse_n_vec, jnp.cross(rel_cpsA, -impulse_n_vec)], axis=-1)
        response_dA = jnp.concatenate([-impulse_d_vec, jnp.cross(rel_cpsA, -impulse_d_vec)], axis=-1)
        responseA = response_nA * apply_n + response_dA * apply_d
        responseA = jnp.sum(responseA, axis=-2) 
        return responseA, response # AB
    else:
        return response

def dynamics_step_fori(states, sim_params:SimParams, substep, jkey, funcs, plane=True):
    '''
    states : (pos, quat, v, w, z, fix_idx)
    '''
    nb, no = nb_no_from_states(states)
    gravity = jnp.array([0.,0.,-9.81])
    # mass = sim_params[2]
    # inertia = sim_params[3]

    # broad phase
    idx1, idx2 = culling(states, plane=False)
    nc = idx1.shape[-1]

    sub_dt = sim_params.dt/float(substep)
    # for ss in range(substep):
    def body_func(ss, carry, init=False):
        if init:
            states, _, _ = carry
        else:
            states, (response, rel_cps_pln , cns_pln, ic_values_pln, const_pln, rel_cps2, cns2, ic_values, const2), _ = carry
        pos, quat, v, w, obj_variables, fix_idx = states
        if plane:
            # plane response
            if init:
                const_pln = get_const_pln(states, jkey, funcs)
            
            ic_values_pln = funcs['pln'](states[0:2], *states[4][:2], jkey, states[4][2])

            r_bar = jnp.cross(const_pln[...,:3], const_pln[...,3:])
            r_bar_mag = jnp.linalg.norm(const_pln[...,3:], axis=-1, keepdims=True) / (1e-6+jnp.linalg.norm(const_pln[...,:3], axis=-1, keepdims=True))
            r_bar = jnp.clip(r_bar_mag, -0.200, 0.200) * tutil.normalize(r_bar)
            cns_pln = tutil.normalize(const_pln[...,:3])
            # plane rel cps len
            len_tp_cp = -(r_bar[...,2] + pos[...,None,2])/ (cns_pln[...,2] + 1e-6)
            rel_cps_pln = r_bar + len_tp_cp[...,None]*cns_pln
            
            # # cp vels
            response_pln = cal_response(rel_v=v, rel_w=w, 
                        rel_cps=rel_cps_pln, cns=cns_pln, ic_values=ic_values_pln, const=const_pln, sim_params=sim_params)
        else:
            response_pln = 0
            rel_cps_pln , cns_pln, ic_values_pln, const_pln = None, None, None, None

        ## object relations
        if no != 1:
            # objects relations
            states1 = jax.tree_util.tree_map(lambda x : global_to_cull_idx(x, idx1), states)
            states2 = jax.tree_util.tree_map(lambda x : global_to_cull_idx(x, idx2), states)

            # narrow phase
            states12 = jax.tree_util.tree_map(lambda *x : jnp.stack(x, axis=2), states1, states2)
            ic_values = funcs['dec'](states12[:2], *states12[4][:2], jkey, states12[4][2])
            if init:
                const2 = get_const(states12, jkey, funcs)

            # rel vels
            rel_vel = states2[2] - states1[2]
            rel_w = states2[3] - states1[3]
            rel_pos = states2[0] - states1[0]

            # cp estimations
            r_bar = jnp.cross(const2[...,:3], const2[...,3:])
            r_bar_mag = jnp.linalg.norm(const2[...,3:], axis=-1, keepdims=True) / (1e-6+jnp.linalg.norm(const2[...,:3], axis=-1, keepdims=True))
            r_bar = jnp.clip(r_bar_mag, -0.080, 0.080) * tutil.normalize(r_bar)
            cns2 = tutil.normalize(const2[...,:3])
            ## aabb contacts end
            l1 = 0.030
            rel_cps2 = r_bar - l1*cns2
            
            response1, response2 = cal_response(rel_v=rel_vel, rel_w=rel_w, 
                    rel_cps=rel_cps2, cns=cns2, ic_values=ic_values, const=const2,
                    sim_params=sim_params, rel_pos=rel_pos)

            # recover origin index
            response_mat = jnp.zeros((nb, no, no, 6))
            bidx = einops.repeat(jnp.arange(nb), 'i -> i j', j=nc)
            # response_mat = response_mat.at[bidx, idx1, idx2].set(-response2)
            response_mat = response_mat.at[bidx, idx1, idx2].set(response1)
            response_mat = response_mat.at[bidx, idx2, idx1].set(response2)
            response_rel = jnp.sum(response_mat, axis=-2)
        else:
            response_rel = 0
            rel_cps2, cns2, ic_values, const2 = None, None, None, None

        drag_wrench = jnp.zeros_like(response_pln)
        drag_wrench = drag_wrench.at[...,:3].set(-states[2] * sub_dt * sim_params.drag_v)
        drag_wrench = drag_wrench.at[...,3:].set(-states[3] * sub_dt * sim_params.drag_w)

        response = response_rel + response_pln + drag_wrench

        # integration
        response_to, response_ao = jnp.split(response, 2, axis=-1)
        # drag force
        otvel = states[2] + sub_dt*gravity + response_to/sim_params.mass
        oavel = states[3] + sub_dt * jnp.einsum('...ij,...j->...i', jnp.linalg.inv(sim_params.inertia), (response_ao/sub_dt-(jnp.cross(states[3], jnp.einsum('...ij,...j->...i',sim_params.inertia,states[3])))))
        opos = states[0] + sub_dt*otvel
        oquat = states[1] + sub_dt * 0.5 * tutil.qmulti(jnp.concatenate([oavel, jnp.zeros_like(oavel[...,:1])], axis=-1), states[1])
        oquat = oquat / (jnp.linalg.norm(oquat, axis=-1, keepdims=True) + 1e-6)
        
        fix_idx = states[5][...,None]
        opos = jnp.where(fix_idx, states[0], opos)
        oquat = jnp.where(fix_idx, states[1], oquat)
        otvel = jnp.where(fix_idx, states[2], otvel)
        oavel = jnp.where(fix_idx, states[3], oavel)
        
        states = (opos, oquat, otvel, oavel, *states[4:])
        
        return (states, (response, rel_cps_pln , cns_pln, ic_values_pln, const_pln, rel_cps2, cns2, ic_values, const2), {})

    states, dyn_infos, metric = jax.lax.fori_loop(1, substep, body_func, body_func(0, (states, None, None), init=True))
    return states, dyn_infos