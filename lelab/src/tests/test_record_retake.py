from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('phase', ['preparing', 'resetting', 'saving', 'completed'])
def test_retake_rejects_nonrecording_phase_without_changing_events(monkeypatch, phase):
    from lelab import record

    events = {'rerecord_episode': False, 'exit_early': False, 'stop_recording': False}
    monkeypatch.setattr(record, 'recording_active', True)
    monkeypatch.setattr(record, 'recording_events', events)
    monkeypatch.setattr(record, 'current_phase', phase)
    assert not record.handle_rerecord_episode()['success']
    assert not any(events.values())


def test_retake_api_marks_current_unsaved_episode_only(client, monkeypatch):
    from lelab import record

    events = {'rerecord_episode': False, 'exit_early': False, 'stop_recording': False}
    monkeypatch.setattr(record, 'recording_active', True)
    monkeypatch.setattr(record, 'recording_events', events)
    monkeypatch.setattr(record, 'current_phase', 'recording')
    monkeypatch.setattr(record, 'current_episode', 3)
    monkeypatch.setattr(record, 'saved_episodes', 2)
    response = client.post('/recording-rerecord-episode')
    assert response.status_code == 200 and response.json()['success']
    assert events['rerecord_episode'] and events['exit_early']
    assert not events['stop_recording']
    assert record.current_episode == 3 and record.saved_episodes == 2


def test_real_recording_branch_discards_buffer_and_retries_same_episode(monkeypatch, tmp_path):
    """Execute the actual LeLab outer recording loop, replacing all hardware I/O."""
    from lelab import record
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101LeaderConfig
    from lerobot.scripts.lerobot_record import RecordConfig
    import lerobot.robots
    import lerobot.teleoperators
    import lerobot.scripts.lerobot_record as recording_script
    import lerobot.utils.feature_utils
    import lerobot.utils.utils

    class Dataset:
        def __init__(self):
            self.buffer = []
            self.saved = [['previously-saved']]
            self.discarded = []
            self.finalized = False

        def clear_episode_buffer(self):
            self.discarded.append(self.buffer[:])
            self.buffer.clear()

        def save_episode(self):
            self.saved.append(self.buffer[:])
            self.buffer.clear()

        def finalize(self):
            self.finalized = True

    dataset = Dataset()
    robot = Mock(cameras={}, action_features={}, observation_features={}, calibration={})
    robot.name = 'so_follower'
    monkeypatch.setattr(lerobot.robots, 'make_robot_from_config', lambda cfg: robot)
    monkeypatch.setattr(lerobot.teleoperators, 'make_teleoperator_from_config', lambda cfg: Mock())
    monkeypatch.setattr(record.LeRobotDataset, 'create', lambda *a, **kw: dataset)
    monkeypatch.setattr(lerobot.utils.feature_utils, 'hw_to_dataset_features', lambda *a, **kw: {})
    monkeypatch.setattr(lerobot.utils.utils, 'log_say', lambda *a, **kw: None)
    monkeypatch.setattr(record, 'safe_disconnect_device', lambda *a, **kw: None)
    events = {'rerecord_episode': False, 'exit_early': False, 'stop_recording': False}
    monkeypatch.setattr(record, 'recording_active', True)
    monkeypatch.setattr(record, 'recording_events', events)
    for name, value in [('current_phase', 'preparing'), ('current_episode', 1), ('saved_episodes', 0), ('current_robot', None), ('phase_start_time', None)]:
        monkeypatch.setattr(record, name, value)
    observed_episodes = []
    reset_count = 0

    def loop(**kwargs):
        nonlocal reset_count
        if kwargs.get('dataset') is None:
            reset_count += 1
            return
        observed_episodes.append(record.current_episode)
        if len(observed_episodes) == 1:
            dataset.buffer.append('bad-take')
            assert record.handle_rerecord_episode()['success']
        else:
            dataset.buffer.append('good-take')
            assert record.handle_exit_early()['success']

    monkeypatch.setattr(recording_script, 'record_loop', loop)
    cfg = RecordConfig(
        robot=SO101FollowerConfig(port='/dev/null', id='offline'),
        teleop=SO101LeaderConfig(port='/dev/null', id='offline-leader'),
        dataset=DatasetRecordConfig(repo_id='local/retake-test', root=tmp_path / 'dataset',
                                    single_task='offline test', num_episodes=1,
                                    episode_time_s=10, reset_time_s=1, video=False, push_to_hub=False),
        play_sounds=False, display_data=False,
    )
    assert record.record_with_web_events(cfg, events) is dataset
    assert observed_episodes == [1, 1]
    assert dataset.discarded == [['bad-take']]
    assert dataset.saved == [['previously-saved'], ['good-take']]
    assert record.saved_episodes == 1 and record.current_episode == 2
    assert reset_count == 2 and dataset.finalized
