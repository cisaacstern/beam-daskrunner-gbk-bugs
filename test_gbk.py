import json
import pathlib
import secrets
from typing import Literal
from unittest.mock import patch

import apache_beam as beam
import dask.bag as db
import pytest
from apache_beam.runners.dask.dask_runner import DaskRunner, TRANSLATIONS
from apache_beam.runners.dask.overrides import _Create
from apache_beam.runners.dask.transform_evaluator import Create as DaskCreate, OpInput
from apache_beam.testing import test_pipeline
from apache_beam.testing.util import assert_that, equal_to

# We will run each of the below tests against both Direct and Dask runners,
# with the DirectRunner serving as a control case of expected behavior
runners = ["DirectRunner", DaskRunner()]
runner_ids = ["DirectRunner", "DaskRunner"]

# Three collections of keyed data to initialize the pipelines with.
# Each uses a different dtype as the key, and each is also paired with
# the expected output after a GBK transform has been applied to it.
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
collections = [str_keys, int_keys, none_keys]
ids = ["str_keys", "int_keys", "none_keys"]


class UnpartitionedCreate(DaskCreate):
    """The DaskRunner GBK bug(s) demonstrated by this test module are (somehow)
    related to the partitioning of the dask bag collection. The following
    object is mocked into the DaskRunner in certain test cases below to
    demonstrate that if the dask bag collection is unpartitioned (i.e., consists
    of only a single partition), then the GBK bug(s) are resolved.
    """
    def apply(self, input_bag: OpInput) -> db.Bag:
        partitioned = super().apply(input_bag)
        return partitioned.repartition(npartitions=1)


# For mocking
TRANSLATIONS_WITH_UNPARTITIONED_DASK_BAG = TRANSLATIONS.copy()
TRANSLATIONS_WITH_UNPARTITIONED_DASK_BAG[_Create] = UnpartitionedCreate


class JSONSerializedAsserter:
    """The bug(s) that this test module aims to demonstrate in the DaskRunner's
    GBK implementation present some interesting testing challenges, namely:

      1. Beam's `assert_that` testing utility itself relies on GBK internally,
         so we can't use `assert_that` to make any assertions about the behavior
         of GBK on DaskRunner. This means we need some kind of custom asserter.
         For simplicity, the custom asserter will only assert against a global
         windowed view of the PCollection.
      2. Two obvious ways to gather the global window are not available to us:
         (a) Side Inputs are not implemented in the DaskRunner, so we can't use
         `apache_beam.pvalue.AsList`; (b) Combiners are also not yet implemented
         in the DaskRunner, so we can't use any type of combiner.

    This object serves as a workaround for this situation, providing methods to
    serialize PCollection values elementwise to JSON, and then retrieve and
    deserialize those values after pipeline completion, thus crudely replicating
    the functionality of `assert_that` for some simple cases.

    Example:
        ```
        tmpdir = ...  # a temporary directory
        expected = ...  # expected global window of PCollection (as list)
        asserter = JSONSerializedAsserter(tmpdir)
        with beam.Pipeline() as p:
            pcoll = p | beam.Create([...])
            pcoll | beam.Map(asserter.serialize_pcoll_elementwise_to_json)

        asserter.assert_against_json_serialized_pcoll(expected)
        ```
    """

    def __init__(self, tmpdir: pathlib.Path):
        self.tmpdir = tmpdir

    def serialize_pcoll_elementwise_to_json(self, element):
        """Intended to be passed to `beam.Map`. Each element of the PCollection
        passed into this method is serialized to a JSON file (with a randomly
        generated name), and stored in the `self.tmpdir` attribute. These
        serialized values can subsequently be retrieved and reassembled into a global
        windowed view of the PCollection by `self.assert_against_json_serialized_pcoll`.
        """
        with (self.tmpdir / f"{secrets.token_hex(16)}.json").open(mode="w") as f:
            json.dump(element, f)

    def assert_against_json_serialized_pcoll(
        self,
        expected: list,
        tuple_elements: bool = True,
    ):
        """Reassemble a PCollection serialized via `self.serialize_pcoll_elementwise_to_json`
        and compare it to the `expected` list. The `tuple_elements` is just a hack to handle
        decoding variable datatypes: if the PCollection elements are expected to be tuples,
        this arguments should keep the default `True` value; otherwise, it should be `False`.
        """
        global_window = []
        for fname in self.tmpdir.iterdir():
            with open(fname) as f:
                element = tuple(json.load(f)) if tuple_elements else json.load(f)
                global_window.append(element)

        assert len(global_window) == len(expected)
        assert sorted(global_window) == expected


@pytest.fixture
def tmpdir(tmp_path_factory: pytest.TempPathFactory):
    yield tmp_path_factory.mktemp("tmp")


@pytest.mark.parametrize("collection, expected", collections, ids=ids)
@pytest.mark.parametrize("runner", runners, ids=runner_ids)
def test_gbk_as_released(
    runner: Literal["DirectRunner"] | DaskRunner,
    collection: list[tuple],
    expected: list[tuple],
    tmpdir: pathlib.Path,
):
    """Test GBK as released in Beam 2.51.0, on both Direct and Dask runners."""
    asserter = JSONSerializedAsserter(tmpdir)
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create(collection) | beam.GroupByKey()
        pcoll | beam.Map(asserter.serialize_pcoll_elementwise_to_json)

    asserter.assert_against_json_serialized_pcoll(expected)



@pytest.mark.parametrize("collection, expected", collections, ids=ids)
@pytest.mark.parametrize("runner", runners, ids=runner_ids)
@patch(
    "apache_beam.runners.dask.dask_runner.TRANSLATIONS",
    TRANSLATIONS_WITH_UNPARTITIONED_DASK_BAG,
)
def test_gbk_with_unpartitioned_dask_bag(
    runner: Literal["DirectRunner"] | DaskRunner,
    collection: list[tuple],
    expected: list[tuple],
    tmpdir: pathlib.Path,
):
    """Patch the DaskRunner's `Create` implementation so that it returns an
    unpartitioned collection, and then test GBK again.
    """
    asserter = JSONSerializedAsserter(tmpdir)
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create(collection) | beam.GroupByKey()
        pcoll | beam.Map(asserter.serialize_pcoll_elementwise_to_json)

    asserter.assert_against_json_serialized_pcoll(expected)



@pytest.mark.parametrize("runner", runners, ids=runner_ids)
def test_beam_create(
    runner: Literal["DirectRunner"] | DaskRunner,
    tmpdir: pathlib.Path,
):
    """Test `beam.Create` on both Direct and Dask runners. This serves as a
    reference test to ensure proper functionality of `JSONSerializedAsserter`,
    and also to prove to ourselves that `beam.Create` does work on both Direct
    and Dask runners.
    """
    four_ints = [0, 1, 2, 3]
    asserter = JSONSerializedAsserter(tmpdir)

    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create(four_ints)
        pcoll | beam.Map(asserter.serialize_pcoll_elementwise_to_json)

    with pytest.raises(AssertionError):
        # if the asserter works correctly, it should actually raise an
        # AssertionError when given a non-matching value for `expected`
        asserter.assert_against_json_serialized_pcoll(
            expected=four_ints[:3],
            tuple_elements=False,
        )
    # and *not* raise an error when given the correct value for `expected`
    asserter.assert_against_json_serialized_pcoll(expected=four_ints, tuple_elements=False)


@pytest.mark.parametrize("runner", runners, ids=runner_ids)
def test_assert_that_as_released(runner):
    """Test `assert_that` testing utility on both Direct and Dask runners."""
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create([0, 1, 2, 3])
        assert_that(pcoll, equal_to([0, 1, 2, 3]))


@pytest.mark.parametrize("runner", runners, ids=runner_ids)
@patch(
    "apache_beam.runners.dask.dask_runner.TRANSLATIONS",
    TRANSLATIONS_WITH_UNPARTITIONED_DASK_BAG,
)
def test_assert_that_with_unpartitioned_dask_bag(runner: Literal["DirectRunner"] | DaskRunner):
    """Patch the DaskRunner's `Create` implementation so that it returns an
    unpartitioned collection, and then test `assert_that` again.
    """
    with test_pipeline.TestPipeline(runner=runner) as p:
        pcoll = p | beam.Create([0, 1, 2, 3])
        assert_that(pcoll, equal_to([0, 1, 2, 3]))


# TODO: show that partitioned groupby *does* work in vanilla dask.bag, so this is a problem
# with the DaskRunner implementation of dask.bag, not dask.bag itself.
