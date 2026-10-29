from __future__ import annotations

import asyncio

import aic_sdk as aic
import gradium
import numpy as np

from .settings import Settings


async def gradium_usage_check(settings: Settings) -> dict:
    client = gradium.client.GradiumClient(api_key=settings.gradium_api_key)
    info = await gradium.usages.get(client)
    return {
        "provider": "gradium",
        "status": "ok",
        "usage_keys": sorted(info.keys()) if isinstance(info, dict) else [],
    }


def ai_coustics_check(settings: Settings) -> dict:
    settings.ai_coustics_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = aic.Model.download(
        settings.ai_coustics_model_id,
        str(settings.ai_coustics_model_dir),
    )
    model = aic.Model.from_file(model_path)
    config = aic.ProcessorConfig.optimal(model, num_channels=1)
    processor = aic.Processor(model, settings.ai_coustics_api_key, config)
    audio = np.zeros((config.num_channels, config.num_frames), dtype=np.float32)
    processed = processor.process(audio)
    return {
        "provider": "ai-coustics",
        "status": "ok",
        "model_id": settings.ai_coustics_model_id,
        "model_path": model_path,
        "sample_rate": config.sample_rate,
        "num_frames": config.num_frames,
        "processed_shape": list(processed.shape),
    }


def run_provider_checks(settings: Settings) -> dict:
    gradium_status = asyncio.run(gradium_usage_check(settings))
    ai_coustics_status = ai_coustics_check(settings)
    return {
        "gradium": gradium_status,
        "ai_coustics": ai_coustics_status,
    }
