"""


"""
import math
import numpy as np
import torch


class AggregateBuffer:
    """
    Attributes
        num_queries (int): Number of predicted actions.
        state_dim (int): State dimension.
        all_time_actions (): A table recording the past `num_queries` predicted
            actions. Roll 1 element in positive direction when an action added.
        action_count (int): Total actions recorded.
    """
    def __init__(
        self,
        num_queries=40,
        state_dim=7,
    ):
        self.num_queries = num_queries
        self.state_dim = state_dim

        self.all_time_actions = torch.zeros((self.num_queries, self.num_queries, self.state_dim)).cuda()
        self.action_count = 0


    def insert_action(self, action):
        """Queue the action to the table"""
        assert len(action.shape) == 3, 'Unsqueeze dim 0 before input'
        assert action.shape[1] == self.num_queries, 'dim 1 of action not equal to self.num_queries'
        assert action.shape[2] == self.state_dim, 'dim 2 of actgion not equal tgo self.state_dim'

        # roll the table
        if self.action_count:
            self.all_time_actions = torch.roll(self.all_time_actions, 1, dims=0)
        self.all_time_actions[[0]] = action  # (1, num_queries, state_dim)
        self.action_count += 1


    def get_aggregate_action(self):
        """Get weighted action """

        # pick all past predicted actions for time t
        actions_for_curr_step = torch.diagonal(self.all_time_actions, dim1=0, dim2=1)  # (state_dim, num_queries)
        actions_for_curr_step = torch.swapaxes(actions_for_curr_step, 0, 1)  # (num_queries, state_dim)

        # consider the initial conditions which slots in `self.all_time_actions` are mostly empty
        if self.action_count < self.num_queries:
            actions_for_curr_step = actions_for_curr_step[:self.action_count]
        actions_for_curr_step = torch.flip(actions_for_curr_step, dims=(0,))  # (num_queries, state_dim)
        assert len(actions_for_curr_step.shape) == 2, 'Shape of `actions_for_curr_step` is unexpected'
        assert actions_for_curr_step.shape[0] <= self.num_queries, 'Size of `actions_for_curr_step` (dim 0) is unexpected'
        assert actions_for_curr_step.shape[1] == self.state_dim, 'Size of `actions_for_curr_step` (dim 1) is unexpected'

        # get exponential weight vector
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()  # (num_queries,)
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)  # (num_queries, 1)
        assert len(exp_weights.shape) == 2, 'Shape of `exp_weights` is unexpected'
        assert exp_weights.shape[0] <= self.num_queries, 'Size of `exp_weights` (dim 0) is unexpected'
        assert exp_weights.shape[1] == 1, 'Size of `exp_weights` (dim 1) is unexpected'

        # get weighted action for time t
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)  # (num_queries, state_dim) -> (1, state_dim)
        assert raw_action.shape == torch.Size([1, self.state_dim]), 'Shape of raw_action is unexpected'

        return raw_action


if __name__=='__main__':
    buffer = AggregateBuffer(
        num_queries=3,
        state_dim=2,
    )
    action1 = np.array([[0.12, 0.11], [0.14, 0.11], [0.13, 0.11]], dtype=np.float32)
    action1 = torch.from_numpy(action1).cuda().unsqueeze(dim=0)  # (1,3,2)
    action2 = np.array([[0.1, 0.11], [0.2, 0.11], [0.3, 0.11]], dtype=np.float32)
    action2 = torch.from_numpy(action2).cuda().unsqueeze(dim=0)
    buffer.set_action(action1)
    buffer.set_action(action2)
    #buffer.set_action(action1)
    print(buffer.all_time_actions, buffer.all_time_actions.shape)
    action = buffer.get_aggregate_action().squeeze(0).cpu().numpy()
    print(action)
