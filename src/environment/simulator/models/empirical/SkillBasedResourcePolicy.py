
from typing import List

import random
from environment.entities.Resource import Resource

from environment.simulator.policies.ResourceAllocationPolicy import ResourceAllocationPolicy



class SkillBasedResourcePolicy(ResourceAllocationPolicy):

    def __init__(self, resources: List[Resource]):
        self.resources = resources

    def select_resource(self, activity, case=None) -> "Resource":
        skilled = [
            r for r in self.resources
            if activity in r.skills
        ]
        if not skilled:
            raise RuntimeError(
                f"No skilled resource for activity {activity.name}"
            )
        return random.choice(skilled)

    def __str__(self):
        return "SkillBasedResourcePolicy"
