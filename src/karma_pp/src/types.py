from dataclasses import dataclass


@dataclass(frozen=True)
class Resolution[OUTCOME]:
    """Base resolution type; mechanisms extend this with additional fields.

    Every resolution always contains the selected_outcome for the agent.
    """

    agent_id: int
    selected_outcome: OUTCOME


@dataclass
class AgentState[PRIVATE_STATE, POLICY_STATE]:
    """State of a single agent."""

    private: PRIVATE_STATE
    policy: POLICY_STATE | None


@dataclass
class PopulationState[AGENT_PRIVATE, POLICY_STATE]:
    """State of the entire population of agents."""

    agent_states: dict[int, AgentState[AGENT_PRIVATE, POLICY_STATE]]

    @property
    def n_agents(self) -> int:
        """Number of agents in the population."""
        return len(self.agent_states)


@dataclass(frozen=True)
class CollectiveAction[OUTCOME, DECISION]:
    """Generic collective action over feasible decisions.

    - agent_ids: agent id for each row (length N_agents), defines canonical agent order
    - decisions: list of feasible decisions
    - signals: per-agent, per-decision signals (shape: N_agents × N_decisions)
    - decisions_to_outcomes: maps decision index to per-agent outcome index (N_decisions × N_agents)
    - agent_outcomes: list of per-agent outcome lists; agent_outcomes[i][k] is
      the kth possible outcome for agent i (length N_agents)
    """

    agent_ids: list[int]
    decisions: list[DECISION]
    signals: list[list[int | float]]
    decisions_to_outcomes: list[list[int]]
    agent_outcomes: list[list[OUTCOME]]
