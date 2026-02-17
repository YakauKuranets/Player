"""
FastAPI server for the PLAYE PhotoLab desktop application.

This module exposes three AI endpoints for enhancing faces, upscaling images
and denoising pictures. Models are loaded at startup using simple wrapper
classes defined in the ``backend/models`` package. Each endpoint accepts an
uploaded file and optional parameters, passes the image through the
corresponding model and returns the processed result as a PNG stream.

The device (CPU or CUDA) is selected automatically depending on the
availability of a GPU. If a model fails to load or run, the endpoint
responds with a 500/503 status and logs the error.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import torch
import numpy as np
from PIL import Image
import io
import logging

# Import model loaders. These modules handle locating and loading the
# PyTorch weights from the models-data directory. They return objects
# with methods appropriate for the inference task (enhance, upscale,
# denoise).
try:
    from models.restoreformer import load_restoreformer
    from models.realesrgan import load_realesrgan
from models.nafnet import load_nafnet
from models.model_paths import get_models_dir
except Exception as exc:
    # If the modules cannot be imported (e.g. missing files), log the
    # error here. Individual loaders will raise when called later.
    logging.error(f"Error importing model loaders: {exc}")


# Configure basic logging to stdout with timestamps. This will make
# debugging easier when running the backend alongside Electron.
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI(title="PLAYE PhotoLab Desktop Backend")

# Models are stored in this dictionary. They are loaded on startup to
# minimise latency for subsequent calls. Keys correspond to endpoint
# names (restoreformer, realesrgan, nafnet).
models = {}

# Determine the target device. If CUDA is available we prefer to use it;
# otherwise we fall back to CPU. The environment variable ``CUDA_VISIBLE_DEVICES``
# can be used to restrict GPUs visible to PyTorch.
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.on_event("startup")
async def startup_event():
    """Load AI models into memory when the server starts."""
    logger.info(f"Starting backend on device: {device}")
    logger.info("Loading AI models...")

    # Attempt to load each model. If a model is missing the loader
    # will raise, which is caught and logged. The endpoint will then
    # return a 503 error when invoked.
    try:
        models['restoreformer'] = load_restoreformer(device)
    except Exception as e:
        logger.error(f"Failed to load RestoreFormer: {e}")
    try:
        models['realesrgan'] = load_realesrgan(device)
    except Exception as e:
        logger.error(f"Failed to load Real-ESRGAN: {e}")
    try:
        models['nafnet'] = load_nafnet(device)
    except Exception as e:
        logger.error(f"Failed to load NAFNet: {e}")

    loaded = [name for name, model in models.items() if model]
    logger.info(f"Models loaded: {loaded}")


@app.get("/health")
async def health_check():
    """Return basic information about the backend's status."""
    return {
        "status": "ok",
        "device": device,
        "models_dir": str(get_models_dir()),
        "models": {k: (v is not None) for k, v in models.items()},
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/ai/face-enhance")
async def enhance_face(file: UploadFile = File(...)):
    """
    Enhance the quality of a face in the uploaded image using RestoreFormer.

    The client sends a multipart/form-data request with a single file
    parameter named ``file``. The response is a PNG image stream.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        model = models.get('restoreformer')
        if not model:
            return JSONResponse(
                status_code=503,
                content={"error": "RestoreFormer model not loaded"}
            )

        result = model.enhance(np.array(image))
        output = Image.fromarray(result)
        buf = io.BytesIO()
        output.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error in face-enhance: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ai/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    factor: int = Form(2)
):
    """
    Upscale the uploaded image by a given factor using Real-ESRGAN.

    The ``factor`` form parameter controls the scaling. Acceptable values
    depend on the model; typical values are 2, 4, 8.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        model = models.get('realesrgan')
        if not model:
            return JSONResponse(
                status_code=503,
                content={"error": "Real-ESRGAN model not loaded"}
            )

        result = model.upscale(np.array(image), scale=factor)
        output = Image.fromarray(result)
        buf = io.BytesIO()
        output.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error in upscale: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ai/denoise")
async def denoise_image(
    file: UploadFile = File(...),
    level: str = Form('medium')
):
    """
    Denoise the uploaded image using NAFNet.

    The ``level`` form parameter may control the strength of denoising.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        model = models.get('nafnet')
        if not model:
            return JSONResponse(
                status_code=503,
                content={"error": "NAFNet model not loaded"}
            )

        result = model.denoise(np.array(image), level=level)
        output = Image.fromarray(result)
        buf = io.BytesIO()
        output.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error in denoise: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    # When executed directly (not under Electron) this runs Uvicorn. When
    # launched by Electron's main process this block is not triggered,
    # because the server is started via subprocess.
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")