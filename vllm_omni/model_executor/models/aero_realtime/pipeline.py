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


_OMNI_PROC = "vllm_omni.model_executor.stage_input_processors.aero_realtime_omni"

AERO_REALTIME_OMNI_PIPELINE = PipelineConfig(
    model_type="aero_realtime_omni",
    model_arch="AeroRealtimeOmniForConditionalGeneration",
    hf_architectures=("AeroRealtimeOmniForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_OMNI_PROC}.thinker2talker_async_chunk",
            custom_process_next_stage_input_func=f"{_OMNI_PROC}.thinker2talker_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_OMNI_PROC}.thinker2talker",
            async_chunk_process_next_stage_input_func=f"{_OMNI_PROC}.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_OMNI_PROC}.talker2code2wav_async_chunk",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],
            },
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="thinker_config",
            engine_output_type="audio",
            custom_process_input_func=f"{_OMNI_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


__all__ = ["AERO_REALTIME_PIPELINE", "AERO_REALTIME_OMNI_PIPELINE"]
