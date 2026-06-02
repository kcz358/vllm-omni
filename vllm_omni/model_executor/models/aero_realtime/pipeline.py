from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)


AERO_REALTIME_PIPELINE = PipelineConfig(
    model_type="aero_realtime",
    model_arch="AeroRealtimeForConditionalGeneration",
    hf_architectures=("AeroRealtimeForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="realtime",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


__all__ = ["AERO_REALTIME_PIPELINE"]
