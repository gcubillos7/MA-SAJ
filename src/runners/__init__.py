REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .masaj_parallel_runner import ParallelRunner
REGISTRY["masaj_parallel"] = ParallelRunner

from .masaj_episode_runner import EpisodeRunner
REGISTRY["masaj"] = EpisodeRunner
