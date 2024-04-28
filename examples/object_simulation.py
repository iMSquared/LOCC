# %%
# import libraries
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import sys, os, inspect
import time
import einops
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import util.model_util as mutil
import util.dynamics_util as dutil
import util.transform_util as tutil
from data_generation import ObjLoad

jkey = jax.random.PRNGKey(0)

# %%
# environment parameters
nb = 4
no = 1
dt = 0.01
substep=4
simstep=700

# %%
# initialize models
base_dir = os.path.join(parentdir, 'saved_model/20220914_084021')
config, funcs  = mutil.model_import(base_dir, pcd_out=False)
dynamics_step_jit = jax.jit(partial(dutil.dynamics_step_fori, funcs=funcs), static_argnames=('substep',))
# %%
# sample point clouds
mesh_dir = os.path.join(currentdir, 'meshes/3D_Dollhouse_Happy_Brother.obj')
objcls = ObjLoad(mesh_dir, o3d_obj=True)
objcls.mesho3d.compute_vertex_normals()
pcd = objcls.mesho3d.sample_points_poisson_disk(number_of_points=1500, init_factor=2)
pcd = (np.asarray(pcd.points), np.asarray(pcd.normals))
pcd = jax.tree_map(lambda x: einops.repeat(x, '... -> r o ...', r=nb, o=no), pcd)

# %%
# convert pcd to z
emb = funcs['enc'](*pcd)
obj_varialbes = (emb[0],emb[2], emb[1])

# %%
# define states
pos = jnp.zeros((nb, no, 3), dtype=jnp.float32)
pos = pos.at[...,0,2].add(0.25)
quat = tutil.qrand((nb,no))
v = jnp.zeros((nb, no, 3), dtype=jnp.float32)
w = jnp.zeros((nb, no, 3), dtype=jnp.float32)
fix_idx = jnp.zeros((nb, no), dtype=jnp.int8)

states = (pos, quat, v, w, obj_varialbes, fix_idx)

# %%
# define sim params
sim_params = dutil.SimParams(dt=dt, friction_coef_pln=0.7, friction_coef_obj=0.7, 
                             drag_v=0.10, drag_w=0.25, rolling_friction_coef=0.02, inertia=np.eye(3)*7e-2, 
                            mass=1, baumgarte_erp_pln=20.0, baumgarte_erp_obj=20.0)


# %%
# start simulation
states, dyn_infos = dynamics_step_jit(states, sim_params, substep, jkey)
state_list = [states]
st = time.time()
for i in range(simstep):
    states, dyn_infos = dynamics_step_jit(states, sim_params, substep, jkey)
    state_list.append(states)
    _, jkey = jax.random.split(jkey)
et =time.time()


# %%
# visualize
import pybullet as p

p.connect(p.GUI)
vm = p.computeViewMatrix(cameraEyePosition=[-0.8,-1.0,1.9], cameraTargetPosition=[-0.1,-0.1,0.0],cameraUpVector=np.array([0,0,1]))
pm = p.getDebugVisualizerCamera()[3]

## entire scene
visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                    halfExtents=[15,15,0.025],
                                    rgbaColor=[100/255., 100/255., 100/255.,1]
                                    )
p.createMultiBody(baseMass=0,
                baseVisualShapeIndex=visualShapeId,
                basePosition=[0,0,-0.425])

pos = np.stack([s[0] for s in state_list], axis=1)
quat = np.stack([s[1] for s in state_list], axis=1)

mb_list_per_env = []
# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

env_edge_no = np.round(nb**(1/2))
tablevisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                halfExtents=[0.48,0.48,0.40],
                                )
for env_idx in range(nb):
    env_offset = np.array([env_idx//env_edge_no - (env_edge_no-1)*0.5, env_idx%env_edge_no - (env_edge_no-1)*0.5, 0]) + np.array([1.5,1.5,0])
    env_offset = env_offset*1.1

    p.createMultiBody(baseMass=0,
                    baseVisualShapeIndex=tablevisualShapeId,
                    basePosition=np.array([0,0,-0.400])+env_offset)

    # make objects
    mb_list = []
    obj_idx = 0
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=mesh_dir,
                                    rgbaColor=[1, 1, 1, 1],
                                    )
    mb_list.append(p.createMultiBody(baseMass=0,
                    baseVisualShapeIndex=visualShapeId,
                    basePosition=pos[env_idx][0][obj_idx]+env_offset,
                    baseOrientation=quat[env_idx][0][obj_idx]))
    mb_list_per_env.append(mb_list)

# %%
# start visualization

intv = int((1/dt) // 20)
for t in range(len(state_list)):
    if t%intv!=0:
        continue
    
    for env_idx in range(nb):
        env_offset = np.array([env_idx//env_edge_no - (env_edge_no-1)*0.5, env_idx%env_edge_no - (env_edge_no-1)*0.5, 0]) + np.array([1.5,1.5,0])
        env_offset = env_offset*1.1
        for i, mb in enumerate(mb_list_per_env[env_idx]):
            p.resetBasePositionAndOrientation(mb, env_offset + pos[env_idx][t][i], quat[env_idx][t][i])
    p.stepSimulation()

    time.sleep(dt*intv)
# %%