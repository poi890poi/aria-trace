"""Explicit learned-route start hints for experimental recorded tracing."""

from contextlib import contextmanager
import json
from pathlib import Path

from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker as Runtime, Pose2D
from benchmarks.localization.reference_cache import identity


def read_start(path, offset=(0.0,0.0)):
    path=Path(path)
    # Only the demonstrated start is inference input; future route states are
    # not read as positions to replay. The complete file hash is provenance.
    with path.open() as stream:
        state=json.loads(stream.readline())
    return {"center_xy":[float(x)+float(d) for x,d in zip(state["canonical_xy"],offset)],
            "mode_id":state["mode_id"],"map_alignment_deg":float(state.get("map_alignment_deg",0)),
            "source":identity(path),"source_state_index":state["state_index"],
            "source_frame_index":state["source_frame_index"],"source_time_ns":state["session_time_ns"],
            "offset_xy":list(offset),"role":"demonstrated-start-prior-not-current-measurement"}


@contextmanager
def installed_start(hint,policy,radius=90.0):
    original_update=Runtime.update
    original_search=Runtime._global_search
    original_global=Runtime._localize_global
    def search(self):
        if self.fusion._state is None:
            return tuple(hint["center_xy"]),radius
        return original_search(self)
    def global_query(self,minimap,mask,yaw_prior,center,search_radius):
        if self.fusion._state is None:
            self.localizer.set_active_mode(hint["mode_id"])
            return self.localizer.localize(minimap,mask,yaw_prior,
                search_center_xy=hint["center_xy"],search_radius_px=radius)
        return original_global(self,minimap,mask,yaw_prior,center,search_radius)
    def update(self,*args,**kwargs):
        if not getattr(self,"_known_start_applied",False):
            self._known_start_applied=True
            if policy=="verified":
                self.set_route_start({"canonical_xy":hint["center_xy"],
                    "mode_id":hint["mode_id"],"map_alignment_deg":hint["map_alignment_deg"],
                    "state_index":hint["source_state_index"]})
            if policy=="prior" and self.fusion._state is None:
                self.fusion.initialize(Pose2D(*hint["center_xy"],hint["map_alignment_deg"]))
                self._activate_map_mode(hint["mode_id"],update_scale=True)
        row=original_update(self,*args,**kwargs)
        row["known_start"]={"policy":policy,"center_xy":hint["center_xy"],
                            "mode_id":hint["mode_id"],"role":hint["role"]}
        return row
    try:
        Runtime.update=update
        Runtime._global_search=search
        Runtime._localize_global=global_query
        yield
    finally:
        Runtime.update=original_update
        Runtime._global_search=original_search
        Runtime._localize_global=original_global
