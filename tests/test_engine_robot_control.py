"""Optional cross-project test: install robot_control in the test env first.

Production model serving never imports robot_control. This test uses a mock adapter.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("xicon_engine")
robot_client = pytest.importorskip("deployment.engine_client")
from robot_control.runtime.adapter import CameraSnapshot
from xicon_engine import PolicyInfo, PolicyServer

from xiconwam.deployment.policy import HoldPolicy


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_wam_fixture_to_real_robot_control_loop(mode):
    writes = []
    adapter = SimpleNamespace(
        state_names=["q0", "q1"],
        camera_keys=[],
        get_joint_state=lambda: np.array([0.25, -0.25]),
        get_eepose_state=lambda: np.zeros(12),
        get_camera_snapshot=lambda: CameraSnapshot(
            images={}, sequences={}, captured_at_s={}
        ),
        reset_pose_reference=lambda: None,
        robot=SimpleNamespace(
            move_to_joint=lambda action: writes.append(action.copy()),
            safe_stop=lambda: None,
        ),
    )
    info = PolicyInfo(
        "hold-test",
        8,
        2,
        2,
        20.0,
        "fixture/v1",
        action_names=("q0", "q1"),
        action_units=("rad", "rad"),
    )

    async def run():
        server = await PolicyServer(HoldPolicy(info), port=0).start()
        try:
            return await asyncio.to_thread(
                robot_client.run_engine_stage,
                adapter,
                policy_url=f"ws://127.0.0.1:{server.port}",
                schema_id=info.schema_id,
                task="fixture",
                fps=20.0,
                mode=mode,
                execute_steps=2,
                chunk_num=2,
                dry_run=False,  # mock adapter only; assert accepted commands
            )
        finally:
            await server.close()

    assert asyncio.run(run()).completed
    assert writes
    np.testing.assert_allclose(writes, np.tile([0.25, -0.25], (len(writes), 1)))
