"""

"""
import os
import numpy as np
import torch
import pickle
import time
from einops import rearrange
from typing import Dict

from gello.agents.agent import Agent
from gello.agents.policy import ACTPolicy, CNNMLPPolicy
from gello.agents.aggregate_buffer import AggregateBuffer


def get_image(obs, camera_names):
    """Extract stacked camera images from observation dictionary.
    Args
        obs (dict): A dictionary contains image observations from different cameras. (with each shape: (480, 640, 3))
        camera_names (List[str]): List of camera names.
    Returns
        (torch.Tensor): Normalized & stacked image observation from different cameras (shape: (1, num_cameras, 3, 480, 640))
    """
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


class ACTAgent(Agent):
    def __init__(
        self,
        policy_path,
        stats_path,
        num_queries,  # TODO: defined by model output shape
        temporal_agg=True,
        sample_period=40,
        num_dofs=6
    ):
        # load policy_config
        self.policy_config = {
            'lr': 1e-5,
            'num_queries': num_queries,
            'kl_weight': 10.,
            'hidden_dim': 512,
            'dim_feedforward': 2048,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['wrist_rgb'],
            'state_dim': 7,
            'dim_feedforward': 3200,
        }
        # set variables
        self.num_dofs = num_dofs
        self.state_dim = 7
        self.policy_path = policy_path
        self.temporal_agg = temporal_agg
        self.num_queries = self.policy_config['num_queries']
        self.sample_period = sample_period  # number of time steps between 2 inferences, ranged from 1 to `num_queries`.
        self.t = 0  # time step
        self.all_actions = None  # holds the inference output, (1, num_queries, 7)

        # set temporal aggregation variables
        if self.temporal_agg:
            self.buffer = AggregateBuffer(
                num_queries=self.num_queries,
                state_dim=self.state_dim,
            )

        # load dataset stat
        self.stats_path = stats_path
        with open(self.stats_path, 'rb') as f:
            stats = pickle.load(f)

        # set pre-process & post-process functions
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        # load policy
        self.policy = ACTPolicy(self.policy_config)
        self.policy.load_state_dict(torch.load(policy_path, weights_only=True))

        # warm-up policy
        with torch.inference_mode():
            self.policy.eval()
            qpos = np.zeros(shape=(7,))
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # (1,7)
            image = np.zeros(shape=(1,3,480,640)) / 255.
            image = torch.from_numpy(image).float().cuda().unsqueeze(0)  # (1,1,3,480,640)
            self.policy(qpos, image)


    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Agent act according to observations """
        if self.t % self.sample_period == 0:
            # pre-process image & qpos
            curr_image = get_image(obs, ["wrist_rgb"])
            qpos = self.pre_process(obs["joint_positions"])
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            
            # inference
            with torch.inference_mode():
                self.policy.eval()
                self.all_actions = self.policy(qpos, curr_image)  # (1,100,7)

        if self.temporal_agg:
            # put inference result to aggregation table
            self.buffer.insert_action(self.all_actions)
            # get weighted action for time t
            raw_action = self.buffer.get_aggregate_action()  # (1,7)
        else:
            raw_action = self.all_actions[:, self.t % self.num_queries]  # (1,7)

        # post-process actions
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = self.post_process(raw_action)  # (7,)

        self.t += 1
        return action


if __name__=='__main__':
    # 20250310_1_ckpts_chunk40, 20250225_1_ckpts_chunk100
    ckpt_dir = '/media/hungyi/Data_sdb5/home/act_ckpts/20250310_1_ckpts_chunk40'
    agent = ACTAgent(policy_path=ckpt_dir+'/policy_last.ckpt', stats_path=ckpt_dir+'/dataset_stats.pkl')
    
    durations = []
    for i in range(100):
        start_t = time.monotonic()
        obs = {
            "joint_positions": np.zeros(shape=(7,)),
            "wrist_rgb": np.zeros(shape=(480,640,3)),
        }
        
        agent.act(obs)
        durations.append(time.monotonic() - start_t)
    print(durations)
    print('Average duration: {} s'.format(np.mean(durations)))
    print('std: {} s'.format(np.std(durations)))
