from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .config import Config
from .cost import CostMeter
from .render import Renderer
from .store import Store


@dataclass
class Backends:
    cfg: Config
    cost: CostMeter
    judge_cost: CostMeter
    planner: Any
    drafter: Any
    layouter: Any
    reviewer: Any
    judge: Any
    generator: Any
    renderer: Renderer
    store: Optional[Store]
    llm: Any = None
    monolithic: Any = None
    runtime: Any = None

    def close(self) -> None:
        self.renderer.close()


def build_backends(cfg: Config, cost: Optional[CostMeter] = None, store: Optional[Store] = None) -> Backends:
    cost = cost or CostMeter(cfg.COST_WEIGHTS)
    judge_cost = CostMeter(cfg.COST_WEIGHTS)
    renderer = Renderer(str(cfg.RENDER_BACKEND), cost=cost, max_side=int(cfg.RENDER_MAX_SIDE))
    if store is None:
        store = Store(str(cfg.STORE_DIR), enabled=bool(cfg.STORE_ENABLED))
    if cfg.mock:
        from .mock import MockDrafter, MockGenerator, MockJudge, MockLayouter, MockMonolithic, MockPlanner, MockReviewer, MockRuntime

        runtime = MockRuntime(seed=int(cfg.MOCK_SEED), error_rate=float(cfg.MOCK_ERROR_RATE), cost=cost)
        planner = MockPlanner(runtime)
        drafter = MockDrafter(runtime)
        layouter = MockLayouter(runtime)
        reviewer = MockReviewer(runtime)
        judge = MockJudge(runtime, judge_cost)
        if str(cfg.GENERATE_BACKEND) == "remote":
            from .generate import RemoteGenerator

            generator: Any = RemoteGenerator(cfg, cost)
        else:
            generator = MockGenerator(runtime)
            generator.mock = True
        monolithic = MockMonolithic(runtime, planner, drafter, layouter, error_scale=float(cfg.MOCK_BASELINE_ERROR_SCALE))
        return Backends(cfg, cost, judge_cost, planner, drafter, layouter, reviewer, judge, generator, renderer, store, llm=None, monolithic=monolithic, runtime=runtime)
    from .executors import LLMDrafter, LLMLayouter
    from .judge import LLMJudge
    from .llm import OpenAICompatible
    from .planner import LLMPlanner
    from .review import LLMReviewer

    llm = OpenAICompatible(cfg, cost)
    judge_llm = OpenAICompatible(cfg, judge_cost)
    planner = LLMPlanner(llm, retries=int(cfg.LLM_RETRIES))
    drafter = LLMDrafter(llm)
    layouter = LLMLayouter(llm)
    reviewer = LLMReviewer(llm, model_name=cfg.role_model("review"))
    judge = LLMJudge(judge_llm)
    if str(cfg.GENERATE_BACKEND) == "remote":
        from .generate import RemoteGenerator

        generator = RemoteGenerator(cfg, cost)
    else:
        from .mock import MockGenerator, MockRuntime

        generator = MockGenerator(MockRuntime(seed=int(cfg.MOCK_SEED)))
        generator.mock = True
    return Backends(cfg, cost, judge_cost, planner, drafter, layouter, reviewer, judge, generator, renderer, store, llm=llm, monolithic=None, runtime=None)
