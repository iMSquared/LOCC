import jax.numpy as jnp
import numpy as np
import jax
import einops

def rand_sphere(outer_shape):
    ext = np.random.normal(size=outer_shape + (5,))
    return (ext / np.linalg.norm(ext, axis=-1, keepdims=True))[...,-3:]


def safe_norm(x, axis, keepdims=False):
    is_zero = jnp.allclose(x, 0.)
    # temporarily swap x with ones if is_zero, then swap back
    x = jnp.where(is_zero, jnp.ones_like(x), x)
    n = jnp.linalg.norm(x, axis=axis, keepdims=keepdims)
    n = jnp.where(is_zero, 0., n)
    return n

# quaternion operations
def normalize(vec):
    return vec/(safe_norm(vec, axis=-1, keepdims=True) + 1e-8)

def quw2wu(quw):
    return jnp.concatenate([quw[...,-1:], quw[...,:3]], axis=-1)

def qrand(outer_shape, jkey=None):
    if jkey is None:
        return qrand_np(outer_shape)
    else:
        return normalize(jax.random.normal(jkey, outer_shape + (4,)))

def qrand_np(outer_shape):
    q1 = np.random.normal(size=outer_shape+(4,))
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    return q1

def line2q(zaxis, yaxis=jnp.array([1,0,0])):
    Rm = line2Rm(zaxis, yaxis)
    return Rm2q(Rm)

def qmulti(q1, q2):
    b,c,d,a = jnp.split(q1, 4, axis=-1)
    f,g,h,e = jnp.split(q2, 4, axis=-1)
    w,x,y,z = a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g, a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e
    return jnp.concatenate([x,y,z,w], axis=-1)

def qinv(q):
    x,y,z,w = jnp.split(q, 4, axis=-1)
    return jnp.concatenate([-x,-y,-z,w], axis=-1)

def sign(q):
    return (q > 0).astype(q.dtype)*2-1

def qlog(q):
    alpha = jnp.arccos(q[...,3:])
    sinalpha = jnp.sin(alpha)
    abssinalpha = jnp.maximum(jnp.abs(sinalpha), 1e-6)
    n = q[...,:3]/(abssinalpha*sign(sinalpha))
    return n*alpha

def q2aa(q):
    return 2*qlog(q)

def qLog(q):
    return qvee(qlog(q))

def qvee(phi):
    return 2*phi[...,:-1]

def qhat(w):
    return jnp.concatenate([w*0.5, jnp.zeros_like(w[...,0:1])], axis=-1)

def aa2q(aa):
    return qexp(aa*0.5)

def q2R(q):
    i,j,k,r = jnp.split(q, 4, axis=-1)
    R1 = jnp.concatenate([1-2*(j**2+k**2), 2*(i*j-k*r), 2*(i*k+j*r)], axis=-1)
    R2 = jnp.concatenate([2*(i*j+k*r), 1-2*(i**2+k**2), 2*(j*k-i*r)], axis=-1)
    R3 = jnp.concatenate([2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i**2+j**2)], axis=-1)
    return jnp.stack([R1,R2,R3], axis=-2)

def qexp(logq):
    if isinstance(logq, np.ndarray):
        alpha = np.linalg.norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = np.maximum(alpha, 1e-6)
        return np.concatenate([logq[...,:3]/alpha*np.sin(alpha), np.cos(alpha)], axis=-1)
    else:
        alpha = safe_norm(logq[...,:3], axis=-1, keepdims=True)
        alpha = jnp.maximum(alpha, 1e-6)
        return jnp.concatenate([logq[...,:3]/alpha*jnp.sin(alpha), jnp.cos(alpha)], axis=-1)

def qExp(w):
    return qexp(qhat(w))

def qaction(quat, pos):
    return qmulti(qmulti(quat, jnp.concatenate([pos, jnp.zeros_like(pos[...,:1])], axis=-1)), qinv(quat))[...,:3]

def qnoise(quat, scale=np.pi*10/180):
    lq = np.random.normal(scale=scale, size=quat[...,:3].shape)
    return qmulti(quat, qexp(lq))

# posquat operations
def pq_inv(pos, quat):
    quat_inv = qinv(quat)
    return -qaction(quat_inv, pos), quat_inv

def pq_action(translate, rotate, pnt):
    return qaction(rotate, pnt) + translate

def pq_multi(pos1, quat1, pos2, quat2):
    return qaction(quat1, pos2)+pos1, qmulti(quat1, quat2)

def pq2H(pos, quat):
    R = q2R(quat)
    return H_from_Rpos(R, pos)

# homogineous transforms
def H_from_Rpos(R, pos):
    H = jnp.zeros(pos.shape[:-1] + (4,4))
    H = H.at[...,-1,-1].set(1)
    H = H.at[...,:3,:3].set(R)
    H = H.at[...,:3,3].set(pos)
    return H

def H_inv(H):
    R = H[...,:3,:3]
    p = H[...,:3, 3:]
    return H_from_Rpos(T(R), (-T(R)@p)[...,0])

# Rm util
def Rm_inv(Rm):
    return T(Rm)

def line2Rm(zaxis, yaxis=jnp.array([1,0,0])):
    zaxis = normalize(zaxis + jnp.array([0,1e-6,0]))
    xaxis = jnp.cross(yaxis, zaxis)
    xaxis = normalize(xaxis)
    yaxis = jnp.cross(zaxis, xaxis)
    Rm = jnp.stack([xaxis, yaxis, zaxis], axis=-1)
    return Rm


def Rm2q(Rm):
    Rm = einops.rearrange(Rm, '... i j -> ... j i')
    con1 = (Rm[...,2,2] < 0) & (Rm[...,0,0] > Rm[...,1,1])
    con2 = (Rm[...,2,2] < 0) & (Rm[...,0,0] <= Rm[...,1,1])
    con3 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] < -Rm[...,1,1])
    con4 = (Rm[...,2,2] >= 0) & (Rm[...,0,0] >= -Rm[...,1,1]) 

    t1 = 1 + Rm[...,0,0] - Rm[...,1,1] - Rm[...,2,2]
    t2 = 1 - Rm[...,0,0] + Rm[...,1,1] - Rm[...,2,2]
    t3 = 1 - Rm[...,0,0] - Rm[...,1,1] + Rm[...,2,2]
    t4 = 1 + Rm[...,0,0] + Rm[...,1,1] + Rm[...,2,2]

    q1 = jnp.stack([t1, Rm[...,0,1]+Rm[...,1,0], Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]-Rm[...,2,1]], axis=-1) / jnp.sqrt(t1)[...,None]
    q2 = jnp.stack([Rm[...,0,1]+Rm[...,1,0], t2, Rm[...,1,2]+Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2]], axis=-1) / jnp.sqrt(t2)[...,None]
    q3 = jnp.stack([Rm[...,2,0]+Rm[...,0,2], Rm[...,1,2]+Rm[...,2,1], t3, Rm[...,0,1]-Rm[...,1,0]], axis=-1) / jnp.sqrt(t3)[...,None]
    q4 = jnp.stack([Rm[...,1,2]-Rm[...,2,1], Rm[...,2,0]-Rm[...,0,2], Rm[...,0,1]-Rm[...,1,0], t4], axis=-1) / jnp.sqrt(t4)[...,None]
 
    q = jnp.zeros(Rm.shape[:-2]+(4,))
    q = jnp.where(con1[...,None], q1, q)
    q = jnp.where(con2[...,None], q2, q)
    q = jnp.where(con3[...,None], q3, q)
    q = jnp.where(con4[...,None], q4, q)
    q *= 0.5

    return q

def pRm_inv(pos, Rm):
    return (-T(Rm)@pos[...,None,:])[...,0], T(Rm)

def pRm_action(pos, Rm, x):
    return (Rm @ x[...,None,:])[...,0] + pos

# 6d utils
def R6d2Rm(x):
    xv, yv = x[...,:3], x[...,3:]
    xv = normalize(xv)
    zv = jnp.cross(xv, yv)
    zv = normalize(zv)
    yv = jnp.cross(zv, xv)
    return jnp.stack([xv,yv,zv], -1)

# 9d utils
def R9d2Rm(x):
    xm = einops.rearrange(x, '... (t i) -> ... t i', t=3)
    u, s, v = jnp.linalg.svd(xm)
    vt = einops.rearrange(v, '... i j -> ... j i')
    det = jnp.linalg.det(jnp.matmul(u,vt))
    vtn = jnp.concatenate([vt[...,:2,:], vt[...,2:,:]*det[...,None,None]], axis=-2)
    return jnp.matmul(u,vtn)


# general
def T(mat):
    return einops.rearrange(mat, '... i j -> ... j i')