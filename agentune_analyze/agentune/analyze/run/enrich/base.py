from abc import ABC, abstractmethod
from collections.abc import Sequence

from duckdb.duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import Dataset, DatasetSink, DatasetSource
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.eval.base import FeatureEvaluator


class EnrichRunner(ABC):
    @abstractmethod
    async def run(self, features: Sequence[Feature], dataset: Dataset,
                  evaluators: Sequence[type[FeatureEvaluator]], conn: DuckDBPyConnection,
                  keep_input_columns: Sequence[str] = (),
                  deduplicate_names: bool = True) -> Dataset:
        """Evaluate the features and return a dataset with a column per feature.

        The output columns are in the same order as the input features, and the column names are
        given by each `feature.name`.

        Args:
            evaluators: used to evaluate features more efficiently.
            keep_input_columns: write these input columns to the output
            deduplicate_names: if some features have the same name and this argument is true,
                               the output column names are deduplicated (using the logic in feature.dedup_names);
                               if it is false, an exception is raised.
        """

    @abstractmethod
    async def run_stream(self, features: Sequence[Feature], dataset_source: DatasetSource,
                         dataset_sink: DatasetSink, evaluators: Sequence[type[FeatureEvaluator]], conn: DuckDBPyConnection,
                         keep_input_columns: Sequence[str] = (),
                         deduplicate_names: bool = True) -> None:
        """Enriches data and writes the results to a sink. May be more efficient than calling `run` in a loop.

        The API does not return a DatasetSource because that is currently hard to implement,
        requiring sync/async bridging.
        """

