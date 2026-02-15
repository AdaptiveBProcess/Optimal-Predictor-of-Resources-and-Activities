from dataclasses import dataclass
from environment.simulator.policies import ProcessingTimePolicy, RoutingPolicy, CalendarPolicy, ArrivalPolicy, ResourceAllocationPolicy, WaitingTImePolicy


@dataclass
class SimulationSetup:
    time_unit: str
    start_timestamp: str
    routing_policy: RoutingPolicy.RoutingPolicy
    processing_time_policy: ProcessingTimePolicy.ProcessingTimePolicy
    waiting_time_policy: WaitingTImePolicy.WaitingTimePolicy
    arrival_policy: ArrivalPolicy.ArrivalPolicy
    calendar_policy: CalendarPolicy.CalendarPolicy
    resource_policy: ResourceAllocationPolicy.ResourceAllocationPolicy
