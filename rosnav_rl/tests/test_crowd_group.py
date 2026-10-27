"""Tests for the nearest pedestrian group input: labels, proximity clusters, space and AGENT_5_GROUPS."""

from __future__ import annotations

import importlib.resources

import arena_people_msgs.msg as arena_people_msgs
import numpy as np
import torch
import yaml

from rosnav_rl.cfg.action_spaces import DifferentialDriveActionSpace
from rosnav_rl.cfg.agent import AgentConfig
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.model.stable_baselines3.cfg import PPO_Cfg, StableBaselinesCfg
from rosnav_rl.model.stable_baselines3.cfg.ppo import PPO_Algorithm_Cfg
from rosnav_rl.observations.data_sources.generators import CrowdGroupClusterGenerator, CrowdGroupGenerator
from rosnav_rl.observations.utils.semantic import nearest_group, proximity_clusters
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.spaces.observation_space.spaces.environment import CrowdGroupSpace
from rosnav_rl.utils.utils import make_mock_env

PARAMS = AgentParameters(robot_radius=0.4)

# a walking group of three ahead, a standing pair further out, a lone pedestrian close by
POSITIONS = np.array([[4.0, 1.0], [4.0, -1.0], [4.0, 0.0], [2.0, 5.0], [2.5, 5.5], [1.0, 0.0]])
VELOCITIES = np.array([[-1.0, 0.0], [-1.2, 0.0], [-1.1, 0.3], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
LABELS = np.array([3, 3, 3, 5, 5, -1])


def _peds(group_ids: list[int]) -> arena_people_msgs.Pedestrians:
    msg = arena_people_msgs.Pedestrians()
    msg.pedestrians = [arena_people_msgs.Pedestrian(id=i, group_id=g) for i, g in enumerate(group_ids)]
    return msg


def test_nearest_group_reads_count_centroid_relative_velocity_and_lateral_spread():
    out = nearest_group(POSITIONS, VELOCITIES, LABELS, np.array([0.5, 0.0, 0.0]), max_range=12.0)

    np.testing.assert_allclose(out, [3.0, 4.0, 0.0, -1.6, 0.1, 2.0], atol=1e-6)


def test_lone_pedestrians_and_groups_out_of_range_are_no_group():
    singles = nearest_group(POSITIONS, VELOCITIES, np.array([0, 1, 2, 3, 4, -1]), np.zeros(3), max_range=12.0)
    far = nearest_group(POSITIONS, VELOCITIES, LABELS, np.zeros(3), max_range=3.0)
    empty = nearest_group(np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=np.int64), np.zeros(3), max_range=12.0)

    assert singles.tolist() == far.tolist() == empty.tolist() == [0.0] * 6


def test_clusters_chain_close_pedestrians_that_walk_alike_and_split_on_disagreeing_velocity():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [6.0, 0.0], [6.5, 0.0]])
    velocities = np.array([[1.0, 0.0], [1.1, 0.0], [1.0, 0.1], [1.0, 0.0], [0.0, 0.0]])

    labels = proximity_clusters(positions, velocities, distance=1.5, velocity_tolerance=0.5)

    assert labels[0] == labels[1] == labels[2]
    assert len({labels[0], labels[3], labels[4]}) == 3


def test_label_generator_groups_by_the_simulator_group_id():
    out = CrowdGroupGenerator(name="crowd_group")._generate(
        _peds(LABELS.tolist()), POSITIONS, VELOCITIES, np.zeros(3), PARAMS
    )

    assert out[0] == 3.0
    np.testing.assert_allclose(out[1:3], [4.0, 0.0], atol=1e-6)


def test_cluster_generator_finds_the_groups_without_labels():
    out = CrowdGroupClusterGenerator(name="crowd_group")._generate(
        _peds([-1] * len(POSITIONS)), POSITIONS, VELOCITIES, np.zeros(3), PARAMS
    )

    assert out[0] == 3.0
    np.testing.assert_allclose(out[1:3], [4.0, 0.0], atol=1e-6)


def test_no_pedestrians_is_no_group():
    out = CrowdGroupGenerator(name="crowd_group")._generate(_peds([]), np.empty(0), np.empty(0), np.zeros(3), PARAMS)

    assert out.tolist() == [0.0] * 6


def test_space_clips_and_normalizes_into_the_unit_box():
    space = CrowdGroupSpace(
        group_count_cap=10, group_max_range=12.0, group_max_velocity=4.0, group_spread_cap=6.0, normalize=True
    )

    encoded = space.encode_observation(np.array([20.0, 12.0, -6.0, 0.0, 8.0, 3.0], dtype=np.float32))

    np.testing.assert_allclose(encoded, [1.0, 1.0, -0.5, 0.0, 1.0, 0.0], atol=1e-6)


def _group_agent() -> RL_Agent:
    return RL_Agent(
        AgentConfig(
            name="group_agent",
            action_space=DifferentialDriveActionSpace(linear_range=(-0.5, 1.0), angular_range=(-1.0, 1.0)),
            parameters=AgentParameters(laser_num_beams=720, robot_radius=0.4),
            framework=StableBaselinesCfg(
                algorithm=PPO_Cfg(architecture_name="AGENT_5_GROUPS", parameters=PPO_Algorithm_Cfg(device="cpu"))
            ),
        )
    )


def test_bundled_observations_provide_what_the_group_agent_requires():
    agent = _group_agent()
    config = yaml.safe_load((importlib.resources.files("rosnav_rl") / "observations" / "observations.yaml").read_text())
    provided = (set(config["datasources"]) - set(config["aliases"].values())) | set(config["aliases"]) | {"is_terminal"}

    required = {
        key for keys in agent.space_manager.observation_space_manager.required_observations.values() for key in keys
    }

    assert required <= provided
    assert config["datasources"][config["aliases"]["crowd_group"]]["type"] == "CrowdGroupGenerator"
    assert config["datasources"]["crowd_group_clusters"]["type"] == "CrowdGroupClusterGenerator"


def test_group_agent_head_reads_the_six_scalars_next_to_the_robot_state():
    agent = _group_agent()
    agent.initialize_model(make_mock_env(ns="", space_manager=agent.space_manager))
    extractor = agent.model.model.policy.features_extractor
    observation = {
        key: np.zeros((1, *space.shape), dtype=np.float32)
        for key, space in agent.space_manager.observation_space.spaces.items()
    }
    grouped = dict(observation, CrowdGroupSpace=np.array([[0.3, 0.4, 0.0, -0.3, 0.0, 0.2]], dtype=np.float32))

    extractor.eval()
    with torch.no_grad():
        alone = extractor({k: torch.as_tensor(v) for k, v in observation.items()})
        near_group = extractor({k: torch.as_tensor(v) for k, v in grouped.items()})

    assert extractor._last_action_size == agent.space_manager.observation_space["LastActionSpace"].shape[-1] + 6
    assert not torch.allclose(alone, near_group)
