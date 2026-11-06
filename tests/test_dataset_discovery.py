import json

import pytest

from xiconwam.data.discovery import discover_dataset_dirs


def leaf(path, version="v3.0"):
    (path / "meta").mkdir(parents=True)
    (path / "meta/info.json").write_text(
        json.dumps({"codebase_version": version, "features": {}})
    )
    return str(path.resolve())


def test_mixed_depth_files_and_leaf_pruning(tmp_path):
    droid = leaf(tmp_path / "Droid_v30")
    task = leaf(tmp_path / "InternData_v30/articulation_tasks/franka/close_laptop")
    coin = leaf(tmp_path / "RoboCOIN/robot/task")
    leaf(tmp_path / "Droid_v30/data/not_a_separate_dataset")
    (tmp_path / "notes.txt").write_text("not a dataset")
    assert discover_dataset_dirs([tmp_path]) == [droid, task, coin]
    assert discover_dataset_dirs([task, droid]) == [task, droid]


def test_alias_overlap_and_cycle_are_deduplicated(tmp_path):
    expected = leaf(tmp_path / "real/task")
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    (tmp_path / "real/loop").symlink_to(tmp_path, target_is_directory=True)
    assert discover_dataset_dirs([tmp_path, tmp_path / "alias", expected]) == [expected]


@pytest.mark.parametrize("version", ["v2.1", None])
def test_unsupported_versions_fail(tmp_path, version):
    leaf(tmp_path, version)
    with pytest.raises(ValueError, match="v3.0"):
        discover_dataset_dirs([tmp_path])


def test_bad_json_and_empty_or_missing_inputs_fail(tmp_path):
    with pytest.raises(ValueError):
        discover_dataset_dirs([])
    with pytest.raises(FileNotFoundError):
        discover_dataset_dirs([tmp_path / "missing"])
    with pytest.raises(FileNotFoundError, match="No LeRobot"):
        discover_dataset_dirs([tmp_path])
    leaf(tmp_path)
    (tmp_path / "meta/info.json").write_text("broken")
    with pytest.raises(ValueError, match="Invalid dataset metadata"):
        discover_dataset_dirs([tmp_path])


def test_each_explicit_root_must_exist_and_contain_data(tmp_path):
    valid = leaf(tmp_path / "valid")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No LeRobot"):
        discover_dataset_dirs([valid, empty])
    with pytest.raises(NotADirectoryError):
        discover_dataset_dirs([tmp_path / "valid/meta/info.json"])


def test_text_precompute_uses_the_same_nested_roots(tmp_path, monkeypatch):
    from xiconwam import precompute_text

    first = leaf(tmp_path / "family/robot/task_a")
    second = leaf(tmp_path / "family/robot/task_b")
    seen = []

    def tasks(path):
        seen.append(str(path))
        return ({0: "pick", 1: "place"}, {})

    monkeypatch.setattr(precompute_text, "load_tasks", tasks)
    prompts = precompute_text.read_unique_prompts([str(tmp_path)])
    assert seen == [first, second]
    assert len(prompts) == 2
