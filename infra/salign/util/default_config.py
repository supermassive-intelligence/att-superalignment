from pydantic import BaseModel


class Config(BaseModel):
    inference_api_url: str = "https://qwen30b.cray-lm.com"
    train_api_url: str = "https://qwen30b.cray-lm.com"
    base_model: str = "qwen"

    target_query_count: int = 32

    query_log_sample_size: int = 3
    question_sample_size: int = 3

    maximum_alternate_queries: int = 2
    maximum_alternate_query_history: int = 32

    maximum_evidence_history: int = 32
    maximum_evidence_queries: int = 5
    maximum_research_queries: int = 2

    maximum_sentences: int = 3

    query_execution_timeout: int = 60
    max_rows_per_query: int = 100

    db_profile_max_length: int = 2048
    create_table_max_length: int = 1536

    max_query_result_length: int = 512
    max_context_length: int = 512

    max_query_refinement_iterations: int = 1

    maximum_reasoners: int = 3
    max_new_reasoners: int = 3
    merged_reasoner_description_limit: int = 8192

    min_test_samples: int = 0

    target_accuracy: float = 1.0

    max_solve_iterations: int = 20

    max_align_iterations: int = 3

    question_variation_count: int = 1

    results_path: str = "infra/salign/data/results"

    trajectories_per_error: int = 3
    maximum_trajectory_history: int = 8

    max_reasoner_training_examples: int = 3

    new_insight_count: int = 2
    maximum_insight_history: int = 16
    maximum_insights: int = 8

    start_over_chance: float = 0.2

    db_adapter: str = "SnowflakeAdapter"
