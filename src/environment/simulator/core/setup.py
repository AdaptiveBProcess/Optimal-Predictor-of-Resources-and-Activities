from dataclasses import dataclass
from environment.simulator.policies import ProcessingTimePolicy, RoutingPolicy, CalendarPolicy, ArrivalPolicy, ResourceAllocationPolicy, WaitingTImePolicy


@dataclass
class SimulationSetup:
    time_unit: str
    start_timestamp: str
    routing_policy: RoutingPolicy.RoutingPolicy
    waiting_time_policy: WaitingTImePolicy.WaitingTimePolicy
    processing_time_policy: ProcessingTimePolicy.ProcessingTimePolicy
    calendar_policy: CalendarPolicy.CalendarPolicy
    arrival_policy: ArrivalPolicy.ArrivalPolicy
    resource_policy: ResourceAllocationPolicy.ResourceAllocationPolicy
