import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOPolicy(nn.Module):

    def __init__(self, state_dim, num_activities, num_resources):
        super().__init__()

        hidden = 256

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Heads
        self.activity_head = nn.Linear(hidden, num_activities)

        # Activity embedding for conditioning
        self.activity_embedding = nn.Embedding(num_activities, 32)

        self.resource_head = nn.Linear(hidden + 32, num_resources)

        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state, activity_mask=None, resource_mask=None):

        features = self.backbone(state)

        # --- Activity ---
        activity_logits = self.activity_head(features)

        if activity_mask is not None:
            activity_logits = activity_logits.masked_fill(
                activity_mask == 0, -1e9
            )

        activity_dist = Categorical(logits=activity_logits)
        activity = activity_dist.sample()

        # --- Resource (conditional) ---
        act_emb = self.activity_embedding(activity)

        res_input = torch.cat([features, act_emb], dim=-1)
        resource_logits = self.resource_head(res_input)

        if resource_mask is not None:
            resource_logits = resource_logits.masked_fill(
                resource_mask == 0, -1e9
            )

        resource_dist = Categorical(logits=resource_logits)
        resource = resource_dist.sample()

        # Log prob
        log_prob = (
            activity_dist.log_prob(activity)
            + resource_dist.log_prob(resource)
        )

        value = self.value_head(features).squeeze(-1)

        return activity, resource, log_prob, value

    def evaluate(self, state, activity, resource,
                 activity_mask=None, resource_mask=None):

        features = self.backbone(state)

        activity_logits = self.activity_head(features)
        if activity_mask is not None:
            activity_logits = activity_logits.masked_fill(
                activity_mask == 0, -1e9
            )

        activity_dist = Categorical(logits=activity_logits)

        act_emb = self.activity_embedding(activity)
        res_input = torch.cat([features, act_emb], dim=-1)

        resource_logits = self.resource_head(res_input)
        if resource_mask is not None:
            resource_logits = resource_logits.masked_fill(
                resource_mask == 0, -1e9
            )

        resource_dist = Categorical(logits=resource_logits)

        log_prob = (
            activity_dist.log_prob(activity)
            + resource_dist.log_prob(resource)
        )

        entropy = (
            activity_dist.entropy()
            + resource_dist.entropy()
        )

        value = self.value_head(features).squeeze(-1)

        return log_prob, entropy, value

    def get_activity_logits(self, state):
        features = self.backbone(state)
        return self.activity_head(features)

    def get_resource_logits(self, state, activity):
        features = self.backbone(state)
        act_emb = self.activity_embedding(activity)
        res_input = torch.cat([features, act_emb], dim=-1)
        return self.resource_head(res_input)

class PPOAgent:
    def __init__(self, state_dim, num_activities, num_resources, device="cpu"):
        self.device = device
        self.policy = PPOPolicy(state_dim, num_activities, num_resources).to(device)
        self.num_activities = num_activities
        self.num_resources = num_resources

    def select_action(self, state, activity_mask, resource_mask_callback, deterministic=True):
        """
        Selects an activity and then a resource using masks.
        resource_mask_callback: a function(activity_idx) -> resource_mask
        """
        self.policy.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_mask_t = torch.FloatTensor(activity_mask).unsqueeze(0).to(self.device)

            # 1. Activity selection
            act_logits = self.policy.get_activity_logits(state_t)
            act_logits = act_logits.masked_fill(act_mask_t == 0, -1e9)
            
            if deterministic:
                activity_idx = torch.argmax(act_logits, dim=-1)
            else:
                activity_idx = Categorical(logits=act_logits).sample()

            # 2. Resource selection (conditional)
            res_mask = resource_mask_callback(activity_idx.item())
            res_mask_t = torch.FloatTensor(res_mask).unsqueeze(0).to(self.device)

            res_logits = self.policy.get_resource_logits(state_t, activity_idx)
            res_logits = res_logits.masked_fill(res_mask_t == 0, -1e9)

            if deterministic:
                resource_idx = torch.argmax(res_logits, dim=-1)
            else:
                resource_idx = Categorical(logits=res_logits).sample()

        return activity_idx.item(), resource_idx.item()
