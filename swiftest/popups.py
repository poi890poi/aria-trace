"""Bounded, explicitly registered interruption handling. No guessed dismissal."""
from dataclasses import dataclass
import time


class PopupHandlingError(RuntimeError):
    pass


@dataclass
class PopupRule:
    name: str
    when: object
    dismiss: object
    max_occurrences: int
    timeout: float
    count: int = 0


def handle_popup(session, observation, deadline, allow_actions):
    # Imported here to keep the shared observation/timeout layer in Session.
    from .session import Observation, AmbiguousTargetError

    active = [(rule, rule.when.matches(observation)) for rule in session.popups.values()]
    active = [(rule, boxes) for rule, boxes in active if boxes]
    if not active:
        return False
    if not allow_actions:
        return True
    if len(active) != 1 or len(active[0][1]) != 1:
        raise AmbiguousTargetError("Multiple pop-ups matched; no dismissal sent.")
    rule, parents = active[0]
    if rule.count >= rule.max_occurrences:
        raise PopupHandlingError("Pop-up occurrence limit reached: " + rule.name)
    limit = min(deadline, time.monotonic() + rule.timeout)
    sent = False
    while True:
        if any(other is not rule and other.when.matches(observation)
               for other in session.popups.values()):
            raise AmbiguousTargetError("Another pop-up appeared during dismissal: " + rule.name)
        parents = rule.when.matches(observation)
        if not parents:
            if sent:
                session.events.append({"popup": rule.name, "status": "dismissed"})
            return True  # Caller must reacquire its intended target.
        if len(parents) != 1:
            raise AmbiguousTargetError("Pop-up bounds became ambiguous: " + rule.name)
        if time.monotonic() >= limit:
            raise PopupHandlingError("Pop-up did not clear within its deadline: " + rule.name)
        if not sent:
            parent = parents[0]
            boxes = rule.dismiss.matches(observation, parent)
            if len(boxes) > 1:
                raise AmbiguousTargetError("Multiple pop-up buttons matched: " + rule.name)
            if boxes:
                box = boxes[0]
                point = rule.dismiss.point(box) if hasattr(rule.dismiss, "point") else box.center
                x, y = (round(v) for v in point)
                area = session._input_area(observation)
                if not (parent.contains(box) and 0 <= x < area.w and 0 <= y < area.h
                        and parent.x <= x < parent.x+parent.w and parent.y <= y < parent.y+parent.h):
                    raise ValueError("Pop-up dismissal must stay inside its matched bounds.")
                if time.monotonic() >= limit:
                    raise PopupHandlingError("Pop-up observation exhausted its deadline: " + rule.name)
                session.device.tap(x, y)
                rule.count += 1
                sent = True
                session.stats["input_events"] += 1
                session.events.append({"popup": rule.name, "touch": [x, y], "occurrence": rule.count})
        session._pause(limit, observation)
        observation = Observation(session)
