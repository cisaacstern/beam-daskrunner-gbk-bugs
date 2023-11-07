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


str_keys = [("a", 0), ("a", 1), ("b", 2), ("b", 3)]
str_keys_gbk_expected = [("a", [0, 1]), ("b", [2, 3])]


# @pytest.mark.parametrize("runner", runners, ids=runner_ids)
# def test_gbk_assert_that(runner):
#     with test_pipeline.TestPipeline(runner=runner) as p:
#         pcoll = p | beam.Create(str_keys) | beam.GroupByKey()
#         assert_that(pcoll, equal_to(str_keys_expected))


@pytest.mark.parametrize("runner", runners, ids=runner_ids)
def test_gbk_plain_assert(
    runner,
    tmp_path_factory: pytest.TempPathFactory,
):
    tmpdir = tmp_path_factory.mktemp("tmp")
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create(str_keys) | beam.GroupByKey()
        pcoll | beam.Map(json_dump, tmpdir=tmpdir)

    aggregate = []
    for fname in tmpdir.iterdir():
        with open(fname) as f:
            aggregate.append(tuple(json.load(f)))
    assert sorted(aggregate) == str_keys_gbk_expected


# @pytest.mark.parametrize("runner", runners, ids=runner_ids)
# def test_int_keys_gbk(runner):
#     int_keys = [(0, 0), (0, 1), (1, 2), (1, 3)]
#     expected = [(0, [0, 1]), (1, [2, 3])]
#     with test_pipeline.TestPipeline(runner=runner) as p:
#         pcoll = p | beam.Create(int_keys) | beam.GroupByKey()
#         assert_that(pcoll, equal_to(expected))
#         pcoll | beam.Map(print)
