"""
FastAPI Server for End-to-End Terrain Generation Pipeline
Provides REST API endpoints for the complete text-to-3D terrain workflow.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import uuid
import json
import os
import torch
import numpy as np
from PIL import Image
import io
import zipfile
import logging
from datetime import datetime
from pathlib import Path

# Local imports
try:
    from clip_encoder import CLIPTextEncoderWithProcessor
    from models_awcgan import CLIPConditionedGenerator
    from remaster_sd_controlnet import TerrainRemaster
    from mesh_visualize import TerrainMeshGenerator, TerrainVisualizer, MeshExporter
except ImportError as e:
    logging.warning(f"Some pipeline modules not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class TerrainGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the terrain")
    negative_prompt: Optional[str] = Field("blurry, low quality, distorted", description="Negative prompt")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    width: int = Field(512, ge=256, le=1024, description="Output width")
    height: int = Field(512, ge=256, le=1024, description="Output height")
    num_inference_steps: int = Field(20, ge=10, le=50, description="Number of diffusion steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    controlnet_scale: float = Field(0.8, ge=0.1, le=1.5, description="ControlNet conditioning strength")
    mesh_method: str = Field("structured_grid", description="Mesh generation method")
    export_formats: List[str] = Field(["ply", "obj"], description="Export formats for 3D mesh")
    enhance_realism: bool = Field(True, description="Whether to apply SD remastering")

class GenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class PipelineConfig:
    """Configuration for the terrain generation pipeline"""
    
    def __init__(self):
        # Model paths and settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = Path("models")
        self.output_dir = Path("Output")
        self.temp_dir = Path("temp")
        
        # Create directories
        for dir_path in [self.models_dir, self.output_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # GAN settings
        self.noise_dim = 128
        self.clip_dim = 512
        self.image_size = 256
        
        # SD settings
        self.sd_model_id = "runwayml/stable-diffusion-v1-5"
        self.controlnet_type = "depth"
        
        # Performance settings for GTX 1650
        self.use_fp16 = True
        self.enable_cpu_offload = True
        self.batch_size = 1

class TerrainPipeline:
    """
    Complete terrain generation pipeline integrating all components.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        
        # Initialize components
        self._init_components()
        
        logger.info(f"TerrainPipeline initialized on {self.device}")
    
    def _init_components(self):
        """Initialize all pipeline components"""
        try:
            # CLIP encoder
            self.clip_encoder = CLIPTextEncoderWithProcessor(
                embedding_dim=self.config.clip_dim,
                device=self.device
            )
            logger.info("CLIP encoder loaded")
            
            # GAN generator (load pre-trained if available)
            self.generator = CLIPConditionedGenerator(
                noise_dim=self.config.noise_dim,
                clip_dim=self.config.clip_dim,
                output_size=self.config.image_size
            ).to(self.device)
            
            # Try to load trained weights
            checkpoint_path = self.config.models_dir / "best_checkpoint.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                logger.info("Loaded pre-trained GAN weights")
            else:
                logger.warning("No pre-trained GAN weights found. Using random initialization.")
            
            self.generator.eval()
            
            # SD remaster (optional)
            try:
                self.remaster = TerrainRemaster(
                    controlnet_type=self.config.controlnet_type,
                    device=self.device,
                    use_fp16=self.config.use_fp16,
                    enable_cpu_offload=self.config.enable_cpu_offload
                )
                logger.info("SD remaster pipeline loaded")
                self.has_remaster = True
            except Exception as e:
                logger.warning(f"SD remaster not available: {e}")
                self.has_remaster = False
            
            # Mesh generator
            self.mesh_generator = TerrainMeshGenerator(method="structured_grid")
            self.visualizer = TerrainVisualizer()
            self.exporter = MeshExporter()
            
            logger.info("All pipeline components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    async def generate_terrain(
        self,
        request: TerrainGenerationRequest,
        task_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Complete terrain generation pipeline.
        
        Args:
            request: Generation parameters
            task_id: Unique task identifier
            progress_callback: Function to update progress
            
        Returns:
            Dict with generated files and metadata
        """
        results = {
            "heightmap_path": None,
            "remastered_path": None,
            "mesh_paths": {},
            "visualization_path": None,
            "metadata": {}
        }
        
        try:
            # Step 1: Generate heightmap with GAN
            if progress_callback:
                await progress_callback(task_id, 0.1, "Encoding text prompt...")
            
            with torch.no_grad():
                # Encode prompt
                clip_embedding = self.clip_encoder.encode_text_with_enhancement(
                    request.prompt, enhance_prompts=True
                )
                
                # Generate noise
                if request.seed is not None:
                    torch.manual_seed(request.seed)
                    np.random.seed(request.seed)
                
                noise = torch.randn(1, self.config.noise_dim, device=self.device)
                
                # Generate heightmap
                if progress_callback:
                    await progress_callback(task_id, 0.3, "Generating base heightmap...")
                
                heightmap_tensor = self.generator(noise, clip_embedding)
                heightmap = heightmap_tensor.cpu().numpy()[0, 0]
                
                # Normalize to [0, 1]
                heightmap = (heightmap + 1) / 2
                heightmap = np.clip(heightmap, 0, 1)
            
            # Save heightmap
            heightmap_path = self.config.output_dir / f"{task_id}_heightmap.png"
            heightmap_img = Image.fromarray((heightmap * 255).astype(np.uint8), mode='L')
            heightmap_img.save(heightmap_path)
            results["heightmap_path"] = str(heightmap_path)
            
            # Step 2: Remaster with Stable Diffusion (optional)
            remastered_image = None
            if request.enhance_realism and self.has_remaster:
                if progress_callback:
                    await progress_callback(task_id, 0.5, "Remastering with Stable Diffusion...")
                
                try:
                    remastered_image, _ = self.remaster.remaster_heightmap(
                        heightmap=heightmap,
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        controlnet_conditioning_scale=request.controlnet_scale,
                        output_size=(request.width, request.height),
                        seed=request.seed
                    )
                    
                    # Save remastered image
                    remastered_path = self.config.output_dir / f"{task_id}_remastered.png"
                    remastered_image.save(remastered_path)
                    results["remastered_path"] = str(remastered_path)
                    
                except Exception as e:
                    logger.warning(f"SD remastering failed: {e}")
                    remastered_image = None
            
            # Step 3: Generate 3D mesh
            if progress_callback:
                await progress_callback(task_id, 0.7, "Generating 3D mesh...")
            
            # Use higher resolution heightmap for mesh if available
            mesh_heightmap = cv2.resize(heightmap, (request.width, request.height)) if heightmap.shape != (request.height, request.width) else heightmap
            
            mesh = self.mesh_generator.generate_mesh(
                heightmap=mesh_heightmap,
                x_scale=1.0,
                y_scale=1.0,
                z_scale=0.3  # Reasonable elevation scaling
            )
            
            # Export mesh in requested formats
            if progress_callback:
                await progress_callback(task_id, 0.8, "Exporting 3D mesh...")
            
            for fmt in request.export_formats:
                mesh_path = self.config.output_dir / f"{task_id}_mesh.{fmt}"
                self.exporter.export_mesh(mesh, str(mesh_path))
                results["mesh_paths"][fmt] = str(mesh_path)
            
            # Step 4: Create visualization
            if progress_callback:
                await progress_callback(task_id, 0.9, "Creating 3D visualization...")
            
            viz_path = self.config.output_dir / f"{task_id}_visualization.png"
            self.visualizer.visualize_terrain(
                mesh=mesh,
                title=f"Terrain: {request.prompt[:50]}...",
                save_path=str(viz_path),
                interactive=False
            )
            results["visualization_path"] = str(viz_path)
            
            # Step 5: Create metadata
            results["metadata"] = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
                "dimensions": {"width": request.width, "height": request.height},
                "generation_params": {
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "controlnet_scale": request.controlnet_scale
                },
                "mesh_method": request.mesh_method,
                "enhanced_realism": request.enhance_realism and self.has_remaster,
                "generated_at": datetime.now().isoformat()
            }
            
            if progress_callback:
                await progress_callback(task_id, 1.0, "Generation completed!")
            
            return results
            
        except Exception as e:
            logger.error(f"Terrain generation failed: {e}")
            raise

# Global pipeline instance
pipeline_config = PipelineConfig()
pipeline = None

# Task management
tasks: Dict[str, TaskStatus] = {}

async def update_task_progress(task_id: str, progress: float, message: str):
    """Update task progress"""
    if task_id in tasks:
        tasks[task_id].progress = progress
        tasks[task_id].message = message
        logger.info(f"Task {task_id}: {progress:.1%} - {message}")

async def process_terrain_generation(task_id: str, request: TerrainGenerationRequest):
    """Background task for terrain generation"""
    try:
        tasks[task_id].status = "processing"
        tasks[task_id].message = "Starting generation..."
        
        # Run generation
        results = await pipeline.generate_terrain(
            request, task_id, 
            progress_callback=update_task_progress
        )
        
        # Update task status
        tasks[task_id].status = "completed"
        tasks[task_id].progress = 1.0
        tasks[task_id].message = "Generation completed successfully"
        tasks[task_id].completed_at = datetime.now().isoformat()
        tasks[task_id].results = results
        
    except Exception as e:
        # Handle errors
        tasks[task_id].status = "failed"
        tasks[task_id].error = str(e)
        tasks[task_id].message = f"Generation failed: {e}"
        logger.error(f"Task {task_id} failed: {e}")

# FastAPI app
app = FastAPI(
    title="Terrain Generation API",
    description="AI-powered text-to-3D terrain generation pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    try:
        pipeline = TerrainPipeline(pipeline_config)
        logger.info("API server started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Terrain Generation API", 
        "version": "1.0.0",
        "status": "running",
        "device": pipeline_config.device if pipeline else "unknown"
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_terrain(
    request: TerrainGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Start terrain generation from text prompt.
    Returns immediately with task ID for polling status.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Task created",
        created_at=datetime.now().isoformat()
    )
    
    # Start background generation
    background_tasks.add_task(process_terrain_generation, task_id, request)
    
    return GenerationResponse(
        task_id=task_id,
        status="pending",
        message="Terrain generation started"
    )

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of a terrain generation task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]

@app.get("/download/{task_id}/{file_type}")
async def download_file(task_id: str, file_type: str):
    """
    Download generated files.
    file_type: 'heightmap', 'remastered', 'mesh_ply', 'mesh_obj', 'visualization', 'all'
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task.status != "completed" or not task.results:
        raise HTTPException(status_code=400, detail="Task not completed or no results")
    
    results = task.results
    
    if file_type == "all":
        # Create zip archive with all files
        zip_path = pipeline_config.output_dir / f"{task_id}_complete.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # Add all generated files
            for key, path in results.items():
                if path and isinstance(path, str) and os.path.exists(path):
                    zip_file.write(path, os.path.basename(path))
                elif key == "mesh_paths" and isinstance(path, dict):
                    for fmt, mesh_path in path.items():
                        if os.path.exists(mesh_path):
                            zip_file.write(mesh_path, os.path.basename(mesh_path))
            
            # Add metadata
            metadata_path = pipeline_config.temp_dir / f"{task_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(results.get("metadata", {}), f, indent=2)
            zip_file.write(metadata_path, "metadata.json")
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"terrain_{task_id}.zip"
        )
    
    else:
        # Return specific file
        file_mapping = {
            "heightmap": results.get("heightmap_path"),
            "remastered": results.get("remastered_path"),
            "visualization": results.get("visualization_path"),
            "mesh_ply": results.get("mesh_paths", {}).get("ply"),
            "mesh_obj": results.get("mesh_paths", {}).get("obj"),
        }
        
        file_path = file_mapping.get(file_type)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {file_type} not found")
        
        return FileResponse(file_path)

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return {"tasks": list(tasks.values())}

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete task and cleanup files"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Cleanup files
    if tasks[task_id].results:
        results = tasks[task_id].results
        for key, path in results.items():
            if isinstance(path, str) and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
            elif key == "mesh_paths" and isinstance(path, dict):
                for mesh_path in path.values():
                    if os.path.exists(mesh_path):
                        try:
                            os.remove(mesh_path)
                        except:
                            pass
    
    # Remove task
    del tasks[task_id]
    
    return {"message": f"Task {task_id} deleted"}

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    info = {
        "device": pipeline_config.device,
        "models_loaded": bool(pipeline),
        "components": {}
    }
    
    if pipeline:
        info["components"] = {
            "clip_encoder": True,
            "gan_generator": True,
            "sd_remaster": pipeline.has_remaster,
            "mesh_generator": True
        }
    
    return info

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable for production
        log_level="info"
    )