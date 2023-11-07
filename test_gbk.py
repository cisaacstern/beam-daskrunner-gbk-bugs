import json
import pathlib
import secrets
from unittest.mock import patch

import apache_beam as beam
import dask.bag as db
import pytest
from apache_beam.runners.dask.dask_runner import DaskRunner, TRANSLATIONS
from apache_beam.runners.dask.overrides import _Create
from apache_beam.runners.dask.transform_evaluator import Create as DaskCreate, OpInput
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that, equal_to

runners = ["DirectRunner", DaskRunner()]
runner_ids = ["DirectRunner", "DaskRunner"]

str_keys = (
    [("a", 0), ("a", 1), ("b", 2), ("b", 3)],
    [("a", [0, 1]), ("b", [2, 3])],
)
int_keys = (
    [(0, 0), (0, 1), (1, 2), (1, 3)],
    [(0, [0, 1]), (1, [2, 3])],
)
none_keys = (
    [(None, 0), (None, 1), (None, 2), (None, 3)],
    [(None, [0, 1, 2, 3])],
)
params = [str_keys, int_keys, none_keys]
ids = ["str_keys", "int_keys", "none_keys"]


class UnpartitionedCreate(DaskCreate):
    def apply(self, input_bag: OpInput) -> db.Bag:
        partitioned = super().apply(input_bag)
        return partitioned.repartition(npartitions=1)
    

TRANSLATIONS_WITH_UNPARTITIONED_DASK_BAG = TRANSLATIONS.copy()
TRANSLATIONS_WITH_UNPARTITIONED_DASK_BAG[_Create] = UnpartitionedCreate


def serialize_pcoll_elementwise_to_json(element, tmpdir: pathlib.Path):
    with (tmpdir / f"{secrets.token_hex(16)}.json").open(mode="w") as f:
        json.dump(element, f)


def assert_against_json_serialized_pcoll(
    tmpdir: pathlib.Path,
    expected: list,
    tuple_elements: bool = True,
):
    global_window = []
    for fname in tmpdir.iterdir():
        with open(fname) as f:
            element = tuple(json.load(f)) if tuple_elements else json.load(f)
            global_window.append(element)

    assert len(global_window) == len(expected)
    assert sorted(global_window) == expected


def _test_gbk(runner, collection, expected, tmpdir):
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create(collection) | beam.GroupByKey()
        pcoll | beam.Map(serialize_pcoll_elementwise_to_json, tmpdir=tmpdir)

    assert_against_json_serialized_pcoll(tmpdir, expected)


@pytest.fixture
def tmpdir(tmp_path_factory: pytest.TempPathFactory):
    yield tmp_path_factory.mktemp("tmp")


@pytest.mark.parametrize("collection, expected", params, ids=ids)
@pytest.mark.parametrize("runner", runners, ids=runner_ids)
def test_gbk_as_released(
    runner,
    collection: list[tuple],
    expected: list[tuple],
    tmpdir: pathlib.Path,
):
    _test_gbk(runner, collection, expected, tmpdir)


@pytest.mark.parametrize("collection, expected", params, ids=ids)
@pytest.mark.parametrize("runner", runners, ids=runner_ids)
@patch(
    "apache_beam.runners.dask.dask_runner.TRANSLATIONS",
    TRANSLATIONS_WITH_UNPARTITIONED_DASK_BAG,
)
def test_gbk_with_unpartitioned_dask_bag(
    runner,
    collection: list[tuple],
    expected: list[tuple],
    tmpdir: pathlib.Path,
):
    _test_gbk(runner, collection, expected, tmpdir)


@pytest.mark.parametrize("runner", runners, ids=runner_ids)
def test_beam_create(
    runner,
    tmpdir: pathlib.Path,
):
    four_ints = [0, 1, 2, 3]
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create(four_ints)
        pcoll | beam.Map(serialize_pcoll_elementwise_to_json, tmpdir=tmpdir)

    with pytest.raises(AssertionError):
        assert_against_json_serialized_pcoll(
            tmpdir,
            expected=four_ints[:3],
            tuple_elements=False,
        )

    assert_against_json_serialized_pcoll(tmpdir, expected=four_ints, tuple_elements=False)


@pytest.mark.parametrize("runner", runners, ids=runner_ids)
def test_assert_that_as_released(runner):
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create([0, 1, 2, 3])
        assert_that(pcoll, equal_to([0, 1, 2, 3]))


@pytest.mark.parametrize("runner", runners, ids=runner_ids)
@patch(
    "apache_beam.runners.dask.dask_runner.TRANSLATIONS",
    TRANSLATIONS_WITH_UNPARTITIONED_DASK_BAG,
)
def test_assert_that_with_unpartitioned_dask_bag(runner):
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create([0, 1, 2, 3])
        assert_that(pcoll, equal_to([0, 1, 2, 3]))
