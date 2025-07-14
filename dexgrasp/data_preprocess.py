import os
import os.path as osp
from glob import glob
import sys
sys.path.append('../')
import numpy as np
import yaml
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
# from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex
# from utils.logger import DataLog
import torch
from isaacgym.torch_utils import *
import gc
import time
from multiprocessing import Process
import copy

def worker_run(args, proc_id, npy_list, cfg, cfg_train,sim_params, agent_index, folder_name, save_npy):
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index, npy_list=npy_list)
    seq_actions = task.grasp_seqs
    N, T, D = seq_actions.size()
    all_obs = torch.zeros(task.num_envs, T, 2582)
    gt_all_actions = torch.zeros(task.num_envs, T, 28)
    sim_all_actions = torch.zeros(task.num_envs, T, 28)
    obj_all_state = torch.zeros(task.num_envs, T, 7)

    # start rollout trajs
    for i in range(seq_actions.shape[1]):
        # actions = torch.zeros((task.num_envs, task.num_actions))
        # actions[:, 2] = 1.0
        if i == 0:
            task.reset_buf = torch.ones(task.num_envs, device=task.device, dtype=torch.long)
            task.progress_buf = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
            env.reset()
        else:
            actions = seq_actions[:, i, :]
            env.task.step(actions, i)
            gt_all_actions[:, i - 1, :] = actions.clone()
            sim_all_actions[:, i - 1, :] = env.task.shadow_hand_dof_pos.clone()
            # obj_all_state[:, i-1, :] = env.task.get_object_state()

            # obj_state, hand_state = env.task.get_object_and_hand_state() #CPU运行才能用
        all_obs[:, i, :] = env.task.obs_buf.clone()
        obj_all_state[:, i, :] = env.task.get_object_state()

    # success_idx_all = torch.where(task.successes==1)[0].to(all_obs.device)
    flag = (task.successes==1).to(all_obs.device)
    total_successes=[]
    all_data = []
    for obj_id in np.unique(task.object_idxs):
        save_dict = {}
        id_mask = task.get_obj_idx_mask(obj_id)
        flag_id = flag*id_mask
        success_idx_id = torch.where(flag_id==1)[0].to(all_obs.device)

        save_dict["obs"] = all_obs[success_idx_id].cpu().numpy()
        gt_all_actions_select = gt_all_actions[success_idx_id].cpu()
        # sim_all_actions_select = sim_all_actions[success_idx_id].cpu() #(B,80, 28)
        # obj_all_state_select = obj_all_state[success_idx].cpu() #(B,80,7)

        vis_unscale_actions = unscale(gt_all_actions_select, task.shadow_hand_dof_lower_limits.cpu(),
                                                   task.shadow_hand_dof_upper_limits.cpu())
        save_dict['vis_unscale_actions'] = torch.clamp(vis_unscale_actions, min=-1.0, max=1.0).numpy()

        # save_dict['sim_unscale_actions'] = unscale(sim_all_actions_select, task.shadow_hand_dof_lower_limits.cpu(),
        #                                            task.shadow_hand_dof_upper_limits.cpu())
        # save_dict['sim_unscale_actions'] = torch.clamp(save_dict['sim_unscale_actions'], min=-1.0, max=1.0).numpy()

        if len(success_idx_id)>0:
            # h2o_vec = env.task.get_h2o_vector(sim_all_actions.cpu(), obj_all_state.cpu(), success_idx) #(N,T,21, 3)
            # save_dict['h2o_vec'] = h2o_vec.numpy()

            hand_pcds, hand_joints = env.task.get_seq_hand_pcd(sim_all_actions.cpu(), success_idx_id)  # (N,T,512,3)
            obj_pcds = env.task.get_seq_obj_pcd(obj_all_state.cpu(), success_idx_id,obj_id)  # (N,T,512,3)
            save_dict['hand_pcds'] = hand_pcds.numpy()
            save_dict['obj_pcds'] = obj_pcds.numpy()


        select_idx = success_idx_id.cpu().numpy()
        save_dict['success_idx'] = select_idx
        save_dict['obj_rotmat'] = env.task.obj_trajs_info['obj_rotmat'][select_idx]
        save_dict['obj_scale'] = env.task.obj_trajs_info['obj_scale'][select_idx]
        save_dict['grasp_seqs'] = env.task.obj_trajs_info['grasp_seqs'][select_idx]
        if 'obj_code_idx' in env.task.obj_trajs_info:
            save_dict['obj_code_idx'] = env.task.obj_trajs_info['obj_code_idx'][obj_id]
        all_data.append(save_dict)

        mean_success = torch.mean(task.successes[id_mask])
        total_successes.append(mean_success.data)
        print(f"Success Rate: {mean_success}")

        # save obs and actions
        if save_npy:
            np.save(f"./{folder_name}/{task.object_code_list[obj_id]}.npy", save_dict)

    env.task.clean_sim()
    del env, task
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Process {proc_id} finished!  Mean SR={torch.stack(total_successes).mean()}")
    a=1
    return all_data

def data_preprocess_online(data):
    set_np_formatting()

    args = get_args()
    # args.task = "ShadowHandGraspDexRepIjrr2"
    # args.algo = "ppo1"
    args.seed = 0
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"
    args.logdir = "logs/dexrep_dexgrasp"
    args.headless = True

    cfg, cfg_train, logdir = load_cfg(args)
    if args.num_objs != -1:
        cfg['env']['num_objs'] = args.num_objs
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    agent_index = get_AgentIndex(cfg)
    folder_name = './dataset/train'
    save_npy = False

    cfg['env']['seq_start_rot_uniform'] = False
    all_npy = copy.deepcopy(data)

    n_per_proc = 30
    npy_groups = [all_npy[i:i + n_per_proc] for i in range(0, len(all_npy), n_per_proc)]
    # worker_run(0, npy_groups[0], cfg,cfg_train,sim_params, agent_index, folder_name, val_folder_name)

    all_data = []

    for proc_id, npy_list in enumerate(npy_groups):
        batch_data = worker_run(args, proc_id, npy_list, cfg, cfg_train, sim_params, agent_index, folder_name, save_npy)
        all_data.extend(batch_data)
    return all_data


def run():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    required_keys = {'obs', 'vis_unscale_actions', 'success_idx'}

    for traj_root_path in ['./dataset/train', './dataset/valid']:
        npy_paths = glob(osp.join(traj_root_path, "*.npy"))

        all_npy = []
        for i, npy_path in enumerate(npy_paths):
            obj_code = osp.basename(npy_path)[:-len('.npy')]
            obj_trajs_info = np.load(npy_path, allow_pickle=True).item()
            if required_keys.issubset(obj_trajs_info.keys()):
                print(f"Skipping already processed file: {npy_path}")
                continue
                
            # if len(obj_trajs_info['grasp_seqs']) > 150:
            #     for key, val in obj_trajs_info.items():
            #         obj_trajs_info[key] = val[:150]
            obj_trajs_info['obj_code'] = obj_code
            all_npy.append(obj_trajs_info)

        n_per_proc = 30
        npy_groups = [all_npy[i:i + n_per_proc] for i in range(0, len(all_npy), n_per_proc)]
        save_npy = True

        for proc_id, npy_list in enumerate(npy_groups):
            worker_run(args, proc_id, npy_list, cfg, cfg_train, sim_params,
                       agent_index, traj_root_path, save_npy=save_npy)

    print("Finish ALL !!!!!")



if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    if args.num_objs != -1:
        cfg['env']['num_objs'] = args.num_objs
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    run()
