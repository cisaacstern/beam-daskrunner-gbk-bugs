import json
import pathlib
import secrets

import apache_beam as beam
import pytest
from apache_beam.runners.dask.dask_runner import DaskRunner
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that, equal_to

runners = ["DirectRunner", DaskRunner()]
runner_ids = ["DirectRunner", "DaskRunner"]


def json_dump(element, tmpdir: pathlib.Path):
    with (tmpdir / f"{secrets.token_hex(16)}.json").open(mode="w") as f:
        json.dump(element, f)


@pytest.mark.parametrize(
    "collection, expected_gbk",
    [
        (
            [("a", 0), ("a", 1), ("b", 2), ("b", 3)],
            [("a", [0, 1]), ("b", [2, 3])],
        ),
        (
            [(0, 0), (0, 1), (1, 2), (1, 3)],
            [(0, [0, 1]), (1, [2, 3])],
        ),
    ],
    ids=["str_keys", "int_keys"],
)
@pytest.mark.parametrize("runner", runners, ids=runner_ids)
def test_gbk(
    runner,
    collection: list[tuple],
    expected_gbk: list[tuple],
    tmp_path_factory: pytest.TempPathFactory,
):
    tmpdir = tmp_path_factory.mktemp("tmp")
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create(collection) | beam.GroupByKey()
        pcoll | beam.Map(json_dump, tmpdir=tmpdir)

    global_window = []
    for fname in tmpdir.iterdir():
        with open(fname) as f:
            global_window.append(tuple(json.load(f)))
    assert sorted(global_window) == expected_gbk

