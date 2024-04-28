import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d
import time
import einops

import util.model_util as mutil
import util.transform_util as tutil
from data_generation import make_dataset, make_sdf_dataset

def dataset_rearrangement(data, cut_no=1000, split=10):
    data = jax.tree_util.tree_map(lambda x: np.array(x), data)
    idx = data[1]
    if len(idx.shape) > 1:
        same_w_prior = np.all(idx[:-1] == idx[1:], axis=-1)
    else:
        same_w_prior = idx[:-1] == idx[1:]
    str_idx_bool = np.concatenate([[True],np.logical_not(same_w_prior)], axis=0)
    str_idx = np.where(str_idx_bool)[0]
    data_len = np.concatenate([str_idx[1:], [idx.shape[0]]],axis=0) - str_idx
    assert np.min(data_len) >= cut_no

    gather_idx = einops.repeat(str_idx, 'i -> i j', j=cut_no)
    gather_idx = gather_idx + np.arange(cut_no)
    data_rearranged = jax.tree_util.tree_map(lambda x: x[gather_idx], data)
    data_rearranged = jax.tree_util.tree_map(lambda x: einops.rearrange(x, 'i (p j) ... -> (i p) j ...', p=split), data_rearranged)
    assert np.all(data_rearranged[1] == data_rearranged[1][:,0:1])

    return data_rearranged


# replay buffer
class replay_buffer(object):
    def __init__(self, data_limit, data_type='collision', data_rearranged=1, cutoff=1000, split=10, weighted_sample=False):
        self.data = None
        # self.size = int(1e9)
        self.size = data_limit
        self.initial_weights = 1.1
        self.weighted_sample = weighted_sample
        self.data_type = data_type
        self.data_rearranged = data_rearranged
        self.cutoff=cutoff
        self.split=split

    def push(self, data):
        if self.data_rearranged:
            data = dataset_rearrangement(data, self.cutoff, self.split)
        if self.data is None:
            # self.data = jax.tree_util.tree_map(lambda x : x[:self.size], data)
            self.data = data
            self.data = self.func_to_dataset(lambda x : x[:self.size])
            if self.weighted_sample:
                self.weights = self.initial_weights * np.ones((self.get_size(),))
        else:
            cur_size = data[1].shape[0]
            # self.data = jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis=0)[:self.size], data, self.data)
            self.data = self.stack_data(data)
            if self.weighted_sample:
                self.weights = np.concatenate([self.weights, self.initial_weights * np.ones((cur_size,))], axis=0)

    def stack_data(self, data):
        if len(self.data) == 3:
            if self.data_type == 'collision':
                return ((np.concatenate([data[0][0], self.data[0][0]], axis=0), np.concatenate([data[0][1],self.data[0][1]], axis=0)), 
                            np.concatenate([data[1],self.data[1]], axis=0), np.concatenate([data[2], self.data[2]], axis=0))
            else:
                return (np.concatenate([data[0], self.data[0]], axis=0), np.concatenate([data[1],self.data[1]], axis=0), 
                            np.concatenate([data[2], self.data[2]], axis=0))
        elif len(self.data) == 4:
            return ((np.concatenate([data[0][0], self.data[0][0]], axis=0), np.concatenate([data[0][1],self.data[0][1]], axis=0)), 
                        np.concatenate([data[1],self.data[1]], axis=0), np.concatenate([data[2], self.data[2]], axis=0), np.concatenate([data[3], self.data[3]], axis=0))

    def func_to_dataset(self, func):
        if len(self.data) == 3:
            if self.data_type == 'collision':
                return ((func(self.data[0][0]), func(self.data[0][1])), func(self.data[1]), func(self.data[2]))
            else:
                return (func(self.data[0]), func(self.data[1]), func(self.data[2]))
        elif len(self.data) == 4:
            return ((func(self.data[0][0]), func(self.data[0][1])), func(self.data[1]), func(self.data[2]), func(self.data[3]))
        else:
            raise ValueError

    def sample(self, jkey, size=500000, type='train'):
        if type=='total':
            idx = np.random.permutation(self.get_size())[:size]
            # return jax.tree_util.tree_map(lambda x : x[idx], self.data)
            return self.func_to_dataset(lambda x : x[idx])
        elif type=='train':
            if self.weighted_sample:
                # weighted sample
                weights_train = self.weights[:self.get_train_size()]
                weights_train = weights_train / np.sum(weights_train)
                idx = np.random.choice(np.arange(self.get_train_size()), size=(size,), replace=False, p=weights_train/np.sum(weights_train))
            else:
                # random sample
                idx = np.random.permutation(self.get_train_size())[:size]
            return jax.tree_util.tree_map(lambda x : x[idx], self.data), idx
        elif type=='val':
            idx = np.random.permutation(np.arange(self.get_train_size(), self.get_size()))[:size]
            # return jax.tree_util.tree_map(lambda x : x[idx], self.data)
            return self.func_to_dataset(lambda x : x[idx])

    def shuffle(self, jkey):
        idx = np.random.permutation(self.get_size())
        # idx = jax.random.permutation(jkey, jnp.arange(self.get_size()))
        if self.weighted_sample:
            self.data, self.weights = jax.tree_util.tree_map(lambda x : x[idx], (self.data, self.weights))
        else:
            # self.data = jax.tree_util.tree_map(lambda x : x[idx], self.data)
            self.data = self.func_to_dataset(lambda x : x[idx])

    def data_slice(self, size):
        # self.data = jax.tree_util.tree_map(lambda x : x[:size], self.data)
        self.data = self.func_to_dataset(lambda x : x[:size])

    def get_size(self):
        return self.data[1].shape[0]

    def get_train_size(self):
        return int(self.get_size() * 0.8)

    def get_val_size(self):
        tsize = self.get_size()
        return tsize - int(tsize * 0.8)
    
    def update_weights(self, idx, weights):
        if len(weights.shape) != 1:
            weights = np.squeeze(weights, axis=-1)
        self.weights[idx] = np.clip(weights, 0.01, 1.0)

    def apply_obj_idx_set(self, obj_idx_set):
        idx_where = np.all(np.any(self.data[1][...,None] == obj_idx_set, axis=-1), axis=-1)
        idx = np.where(idx_where)[0]
        self.data = self.func_to_dataset(lambda x : x[idx])

def measure_duration(func_name, func, *args):
    func_jit = jax.jit(func)
    func_jit(*args)
    st = time.time()
    itr = 2000
    for _ in range(itr):
        func_jit(*args)
    print('inf time for {}: {}'.format(func_name, (time.time() - st)/itr))

def init_model(jkey, config, pcd_pnts_list, pcd_normals_list, measure_inf_time=False):
    
    sample_data = make_dataset(2)

    pcd_pnts = pcd_pnts_list[np.ones(sample_data[1].shape, dtype=np.int32)]
    pcd_normals = pcd_normals_list[np.ones(sample_data[1].shape, dtype=np.int32)]
    enc_model = mutil.PCDEncoder(base_dim=config['enc_base_dim'], num_layers=config['enc_num_layers'], voxel_size=config['voxel_size'], 
                                latent_type1=config['latent_type1'], latent_type2=config['latent_type2'], feature_dim=config['feature_dim'],
                                voxel_factor=config['voxel_factor'])

    emb, enc_params = enc_model.init_with_output(jkey, pcd_pnts, pcd_normals)

    dec_model = mutil.DecModel(base_dim=config['dec_base_dim'], num_layers=config['dec_num_layers'], ns=config['ns'], 
                                latent_type1=config['latent_type1'], latent_type2=config['latent_type2'], cls_type=config['cls_type'], local_type=config['local_type'])
    res, dec_params = dec_model.init_with_output(jkey, sample_data[0], emb[0], emb[2], jkey, emb[1])
    if measure_inf_time:
        if config['latent_type1'] == 'discrete':
            measure_duration('dec_model', dec_model.apply, dec_params, sample_data[0], emb[0], emb[2], jkey, None)
        else:
            measure_duration('dec_model', dec_model.apply, dec_params, sample_data[0], emb[0], emb[2], jkey, emb[1])

    ren_model = None
    occ_model = None
    pln_model = None
    ren_params = None
    occ_params = None
    pln_params = None
    if config['train_plane_model'] == 1:
        pln_model = mutil.PlaneModel(num_layers=config['pln_num_layers'], base_dim=config['pln_base_dim'], 
                            cls_type=config['cls_type'], latent_type2=config['latent_type2'], local_type=config['local_type'])
        pln_params = pln_model.init(jkey, sample_data[0], emb[0], emb[2], jkey, emb[1])
    if config['train_render_model'] == 1:
        ren_model = mutil.RenderModel(num_layers=config['ren_num_layers'], base_dim=config['ren_base_dim'], 
                            cls_type=config['cls_type'], latent_type1=config['latent_type1'], latent_type2=config['latent_type2'])
        emb_sole = jax.tree_util.tree_map(lambda x : x[:,0], emb)
        raylen = jnp.ones_like(emb_sole[0][2][...,0])
        ren_res, ren_params = ren_model.init_with_output(jkey, sample_data[0], emb_sole[0], emb_sole[2], jkey, emb_sole[1], raylen)
    if config['train_occ_model'] == 1:
        sample_data = make_sdf_dataset(nitr=1, ns=100)
        pcd_pnts = pcd_pnts_list[sample_data[1]]
        pcd_normals = pcd_normals_list[sample_data[1]]
        emb = enc_model.apply(enc_params, pcd_pnts, pcd_normals)
        occ_model = mutil.OCCModel(num_layers=config['occ_num_layers'], base_dim=config['occ_base_dim'], latent_type1=config['latent_type1'], latent_type2=config['latent_type2'])
        occ_res, occ_params = occ_model.init_with_output(jkey, sample_data[0], emb[0], emb[2], jkey, emb[1])
        if measure_inf_time:
            measure_duration('occ_model', occ_model.apply, occ_params, sample_data[0], emb[0], emb[2], jkey, emb[1])

    models = (enc_model, dec_model, ren_model, occ_model, pln_model)
    params = (enc_params, dec_params, ren_params, occ_params, pln_params)

    return models, params

def gen_loss_func(config, models):

    def loss_func(params, jkey, data, idx_dir, pcd_pnts_list, pcd_normals_list):

        def enc_dec_loss(x, pcd, y_binary, jkey, enc_model, dec_model, enc_params, dec_params, batch_idx, ray_len=None, pln_gt=None):

            emb = enc_model.apply(enc_params, *pcd)
            xz = emb[1]
            if ray_len is not None:
                yp = dec_model.apply(dec_params, x, emb[0], emb[2], jkey, xz, ray_len)
            else:
                yp = dec_model.apply(dec_params, x, emb[0], emb[2], jkey, xz)

            pln_bce_loss_batch = 0.0
            if config['train_plane_model'] == 1 and pln_gt is not None:
                # make labels for plane contact
                _, jkey = jax.random.split(jkey)
                label_h_dist = jax.random.normal(jkey, shape=pln_gt.shape) * 0.040
                height_offset = -pln_gt + label_h_dist
                x_shifted = (x[0].at[...,2].add(height_offset), x[1])

                pln_label = label_h_dist
                pln_label = jnp.clip(pln_label, -0.02, 0.02)/0.02
                pln_label = 0.5*pln_label + 0.5

                pln_model = models[-1]
                pln_params = params[-2]
                pln_yp = pln_model.apply(pln_params, x_shifted, emb[0], emb[2], jkey, xz)
                pln_label = pln_label[...,None]
                if batch_idx is not None:
                    pln_batch_idx = jnp.stack([jnp.zeros_like(batch_idx), batch_idx], axis=-1)
                    pln_yp = jnp.take_along_axis(pln_yp, pln_batch_idx[...,None], axis=-1)

                pln_yp_binary = pln_yp * 0.5 + 0.5
                pln_yp_binary = jnp.clip(pln_yp_binary, 1e-6, 1.-1e-6)
                pln_bce_loss_batch = -pln_label * jnp.log(pln_yp_binary) - (1.-pln_label) * jnp.log(1.-pln_yp_binary)
                pln_bce_loss_batch = jnp.sum(pln_bce_loss_batch, axis=-2)

            if batch_idx is not None:
                yp = jnp.take_along_axis(yp, batch_idx[...,None], axis=-1)

            yp_binary = yp * 0.5 + 0.5
            yp_binary = jnp.clip(yp_binary, 1e-6, 1.-1e-6)
            if len(yp_binary.shape) - len(y_binary.shape)!=0:
                y_binary = y_binary[...,None]
            bce_loss_batch = -y_binary * jnp.log(yp_binary) - (1.-y_binary) * jnp.log(1.-yp_binary)

            reg_loss2 = 0
            if config['latent_type2'] == 'gg':
                reg_loss = jnp.sum(emb[0][2]**2, axis=-1)
            if config['latent_type2'] == 'vv':
                reg_loss = jnp.sum(xz**2, axis=-1)
            # reg_loss = jnp.mean(reg_loss)

            return bce_loss_batch+0.25*pln_bce_loss_batch, reg_loss, reg_loss2

        x, z_idx, y = data[0]
        y_binary = (y>0).astype(jnp.float32)
        y_binary = jnp.where(y == 0, 0.5, y_binary)
        
        if config['data_rearranged'] == 1:
            pcd_pnts = pcd_pnts_list[z_idx[:,0:1]]
            pcd_normals = pcd_normals_list[z_idx[:,0:1]]
        else:
            pcd_pnts = pcd_pnts_list[z_idx]
            pcd_normals = pcd_normals_list[z_idx]

        pln_gt = None
        if config['train_plane_model'] == 1:
            # pln_gt = jnp.any(tutil.pq_action(x[0][...,None,:], x[1][...,None,:], pcd_pnts)[...,2] <= 0, axis=-1).astype(jnp.float32)
            pln_gt = tutil.pq_action(x[0][...,None,:], x[1][...,None,:], pcd_pnts)[...,2]
            pln_gt = jnp.min(pln_gt, axis=-1)
            # pln_gt = jnp.clip(pln_gt, -0.02, 0.02)/0.02
            # pln_gt = 0.5*pln_gt + 0.5

        if config['pcd_rotation'] > 0:
            # random rotation
            if config['data_rearranged'] == 1:
                # random_quat_apply = tutil.qrand(x[1].shape[:-3]+(1,1,), jkey)
                shapes = x[1].shape[:-3]+(1,1,)
            else:
                # random_quat_apply = tutil.qrand(x[1].shape[:-2]+(1,), jkey)
                shapes = x[1].shape[:-2]+(1,)
            if config['pcd_rotation'] == 1:
                random_quat_apply = tutil.qrand(shapes, jkey)
            elif config['pcd_rotation'] == 2:
                random_quat_apply = jnp.array([[0,0,0],
                                                [0.5*np.pi,0,0],[np.pi,0,0],[1.5*np.pi,0,0],
                                                [0,0.5*np.pi,0],[0,np.pi,0],[0,1.5*np.pi,0],
                                                [0,0,0.5*np.pi],[0,0,np.pi],[0,0,1.5*np.pi]])
                random_quat_apply = random_quat_apply[jax.random.randint(jkey, shape=shapes, minval=0, maxval=random_quat_apply.shape[0])]
                random_quat_apply = tutil.qExp(random_quat_apply)
            elif config['pcd_rotation'] == 3:
                random_quat_apply = tutil.qrand(shapes, jkey)
                random_quat_apply = jnp.where(jax.random.randint(jkey,minval=0,maxval=100,shape=random_quat_apply[...,0:1].shape)<25, tutil.qExp(jnp.zeros_like(random_quat_apply)), random_quat_apply)
            x = (x[0], tutil.qmulti(x[1], tutil.qinv(random_quat_apply)))

            pcd_pnts = tutil.qaction(random_quat_apply[...,None,:], pcd_pnts)
            pcd_normals = tutil.qaction(random_quat_apply[...,None,:], pcd_normals)
        pcd = (pcd_pnts, pcd_normals)
        batch_idx = None
        if config['cls_type'] == 1:
            ## add epsilon ##
            pos, quat = x
            # epsilon = jnp.array([0.002,0.002,0.002,np.pi/180.0*3.0,np.pi/180.0*3.0,np.pi/180.0*3.0])
            # epsilon = jnp.array([0.003,0.003,0.003,np.pi/180.0*8.0,np.pi/180.0*8.0,np.pi/180.0*8.0])
            epsilon = jnp.array([0.0015,0.0015,0.0015,np.pi/180.0*5.0,np.pi/180.0*5.0,np.pi/180.0*5.0])
            _, jkey = jax.random.split(jkey)
            batch_idx = jax.random.randint(jkey, shape=z_idx.shape[:-1], minval=0, maxval=13)
            epsilon_dir = epsilon * idx_dir[batch_idx]
            pq2_sifted = tutil.pq_multi(pos[...,1,:], quat[...,1,:], *tutil.pq_inv(epsilon_dir[...,:3], tutil.qExp(epsilon_dir[...,3:])))
            x = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=-2), (pos[...,0,:], quat[...,0,:]), pq2_sifted)
            ## add epsilon ##

        bce_loss_batch, reg_loss, reg_loss2 = enc_dec_loss(x, pcd, y_binary, jkey, models[0], models[1], params[0], params[1], params[-1], batch_idx, pln_gt=pln_gt)
        _, jkey = jax.random.split(jkey)

        if config['train_render_model']:
            ## loss for ray
            x, z_idx, y, ray_len = data[1]

            y_binary = (y>0).astype(jnp.float32)
            y_binary = jnp.where(y == 0, 0.5, y_binary)
            
            if config['data_rearranged'] == 1:
                pcd_pnts = pcd_pnts_list[z_idx[:,0:1,...,0]]
                pcd_normals = pcd_normals_list[z_idx[:,0:1,...,0]]
            else:
                pcd_pnts = pcd_pnts_list[z_idx[...,0]]
                pcd_normals = pcd_normals_list[z_idx[...,0]]
            # pcd_pnts = pcd_pnts_list[z_idx[...,0]]
            # pcd_normals = pcd_normals_list[z_idx[...,0]]

            if config['pcd_rotation'] == 1:
                if config['data_rearranged'] == 1:
                    random_quat_apply = tutil.qrand(x[1].shape[:-3]+(1,1,), jkey)
                else:
                    random_quat_apply = tutil.qrand(x[1].shape[:-2]+(1,), jkey)
                # random_quat_apply = tutil.qrand(x[1].shape[:-2]+(1,), jkey=jkey)
                quat = jnp.concatenate([tutil.qmulti(x[1][...,0:1,:], tutil.qinv(random_quat_apply)), x[1][...,1:2,:]], axis=-2)
                x = (x[0], quat)
                pcd_pnts = tutil.qaction(random_quat_apply, pcd_pnts)
                pcd_normals = tutil.qaction(random_quat_apply, pcd_normals)

            pcd = (pcd_pnts, pcd_normals)
            # batch_idx = None
            # if config['cls_type'] == 1:
            # ray offset
            pos, quat = x
            num_ray_out = 5
            _, jkey = jax.random.split(jkey)
            batch_idx = jax.random.randint(jkey, shape=z_idx.shape[:-1], minval=0, maxval=num_ray_out)
            len_offset = jnp.linspace(-0.5,0.5,num=num_ray_out,endpoint=True)[batch_idx] * ray_len * (num_ray_out*0.5-0.5)
            z_dir = tutil.q2R(quat[...,1,:])[...,:,2]
            npos2 = pos[...,1,:] - z_dir * len_offset[...,None]
            x = (jnp.stack([pos[...,0,:], npos2], axis=-2), quat)
            # ray offset

            bce_loss_batch_ren, reg_loss_ren, reg_loss2_ren = enc_dec_loss(x, pcd, y_binary, jkey, models[0], models[2], params[0], params[2], params[-1], batch_idx, ray_len)
        else:
            bce_loss_batch_ren, reg_loss_ren, reg_loss2_ren =0,0,0

        if config['train_occ_model']:
            ## loss for ray
            x, z_idx, y = data[2]

            y_binary = (y>0).astype(jnp.float32)
            y_binary = jnp.where(y == 0, 0.5, y_binary)
            
            if config['data_rearranged'] == 1:
                pcd_pnts = pcd_pnts_list[z_idx[:,0:1]]
                pcd_normals = pcd_normals_list[z_idx[:,0:1]]
            else:
                pcd_pnts = pcd_pnts_list[z_idx]
                pcd_normals = pcd_normals_list[z_idx]
            pcd = (pcd_pnts, pcd_normals)

            bce_loss_batch_occ, reg_loss_occ, reg_loss2_occ = enc_dec_loss(x, pcd, y_binary, jkey, models[0], models[3], params[0], params[3], params[-1], None)
        else:
            bce_loss_batch_occ, reg_loss_occ, reg_loss2_occ = 0, 0, 0

        return (jnp.mean(bce_loss_batch) + config['reg_loss_scale'] * jnp.mean(reg_loss) + 0.1*jnp.mean(reg_loss2) + 
                0.5*(jnp.mean(bce_loss_batch_ren) + config['reg_loss_scale'] * jnp.mean(reg_loss_ren) + 0.1*jnp.mean(reg_loss2_ren)) + 
                0.25*(jnp.mean(bce_loss_batch_occ) + config['reg_loss_scale'] * jnp.mean(reg_loss_occ) + 0.1*jnp.mean(reg_loss2_occ))), (bce_loss_batch, bce_loss_batch_ren, bce_loss_batch_occ)

    return loss_func