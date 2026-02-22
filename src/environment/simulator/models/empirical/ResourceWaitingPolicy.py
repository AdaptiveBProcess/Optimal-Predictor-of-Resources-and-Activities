from environment.simulator.policies.WaitingTImePolicy import WaitingTimePolicy


class ResourceWaitingPolicy(WaitingTimePolicy):
    def __init__(self, resource_pool):
        self.resources = resource_pool

    def compute_wait(self, now, activity, resource_id):
        if self.resources.is_available(resource_id, now):
            return 0.0
        return self.resources.next_release_time(resource_id) - now
