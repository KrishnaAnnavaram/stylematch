import dataclasses

import pytest

from stylematch.config import Settings
from stylematch.index import build_or_load
from stylematch.recommender import Recommender
from stylematch.synthetic import write_synthetic


@pytest.fixture(scope="session")
def data_paths(tmp_path_factory):
    return write_synthetic(tmp_path_factory.mktemp("data"), n_products=600, n_queries=24, seed=3)


@pytest.fixture(scope="session")
def settings(data_paths, tmp_path_factory):
    return dataclasses.replace(Settings(), catalog=data_paths["catalog"], index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture(scope="session")
def index(settings):
    built, _ = build_or_load(settings.catalog, settings.index_dir, settings)
    return built


@pytest.fixture(scope="session")
def recommender(index, settings):
    return Recommender(index, settings, client=None)
