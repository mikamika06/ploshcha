"""Редьюсер світу: чистий, детермінований. Деталі — docs/sprints/S0-core.md."""

from pydantic import BaseModel

from .action import Action
from .state import Utterance, WorldState, phase_of

UNKNOWN_AGENT = "unknown_agent"
UNKNOWN_POI = "unknown_poi"
POI_NOT_USABLE = "poi_not_usable"
NOT_AT_POI = "not_at_poi"
UNKNOWN_RECIPIENT = "unknown_recipient"
RECIPIENT_NOT_HERE = "recipient_not_here"
SPEAKING_TO_SELF = "speaking_to_self"


class ActionOutcome(BaseModel):
    agent: str
    action_type: str
    ok: bool
    reason: str | None = None


class TickResult(BaseModel):
    state: WorldState
    outcomes: list[ActionOutcome]

    @property
    def rejected(self) -> list[ActionOutcome]:
        return [o for o in self.outcomes if not o.ok]


def validate_action(state: WorldState, agent_id: str, action: Action) -> str | None:
    agent = state.agents.get(agent_id)
    if agent is None:
        return UNKNOWN_AGENT

    match action.type:
        case "move_to":
            if action.poi not in state.pois:
                return UNKNOWN_POI

        case "use_object":
            poi = state.pois.get(action.poi)
            if poi is None:
                return UNKNOWN_POI
            if not poi.usable:
                return POI_NOT_USABLE
            if agent.location != action.poi:
                return NOT_AT_POI

        case "speak":
            for rid in action.to:
                if rid == agent_id:
                    return SPEAKING_TO_SELF
                recipient = state.agents.get(rid)
                if recipient is None:
                    return UNKNOWN_RECIPIENT
                if recipient.location != agent.location:
                    return RECIPIENT_NOT_HERE

        case "post_to_board" | "reflect" | "wait":
            pass

    return None


def _apply(state: WorldState, agent_id: str, action: Action) -> None:
    agent = state.agents[agent_id]

    match action.type:
        case "move_to":
            agent.location = action.poi
            agent.activity = "moving"

        case "use_object":
            agent.activity = f"using:{action.poi}"

        case "speak":
            state.utterances.append(
                Utterance(
                    tick=state.tick,
                    speaker=agent_id,
                    to=list(action.to),
                    text=action.text,
                    poi=agent.location,
                )
            )
            agent.activity = "speaking"

        case "post_to_board":
            state.board.append(action.topic)
            agent.activity = "posting"

        case "reflect":
            agent.activity = "reflecting"

        case "wait":
            agent.activity = "waiting"


def tick(state: WorldState, decisions: dict[str, Action]) -> TickResult:
    new_state = state.model_copy(deep=True)
    outcomes: list[ActionOutcome] = []

    for agent_id in sorted(decisions):
        action = decisions[agent_id]
        reason = validate_action(new_state, agent_id, action)
        if reason is None:
            _apply(new_state, agent_id, action)
        outcomes.append(
            ActionOutcome(agent=agent_id, action_type=action.type, ok=reason is None, reason=reason)
        )

    new_state.tick = state.tick + 1
    new_state.time_of_day = phase_of(new_state.tick)
    return TickResult(state=new_state, outcomes=outcomes)
