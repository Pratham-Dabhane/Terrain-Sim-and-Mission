"""
CLIP Text Encoder Wrapper for Terrain Generation
Converts text prompts into embeddings for conditioning the aWCGAN.
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder wrapper for generating embeddings from text prompts.
    
    Args:
        model_name: CLIP model name (default: openai/clip-vit-base-patch32)
        device: Device to run the model on
        max_length: Maximum token length
        embedding_dim: Output embedding dimension for GAN conditioning
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 77,
        embedding_dim: int = 512,
    ):
        super().__init__()
        
        self.device = device
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Load CLIP tokenizer and text encoder
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name)
            
            # Move to device and set to eval mode
            self.text_encoder = self.text_encoder.to(device)
            self.text_encoder.eval()
            
            # Get actual output dimension from CLIP
            self.clip_dim = self.text_encoder.config.hidden_size
            
            # Projection layer to match desired embedding dimension
            if self.clip_dim != embedding_dim:
                self.projection = nn.Linear(self.clip_dim, embedding_dim).to(device)
            else:
                self.projection = None
            
            logger.info(f"CLIP encoder loaded: {model_name}")
            logger.info(f"CLIP dimension: {self.clip_dim}, Target dimension: {embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def encode_text(
        self, 
        prompts: Union[str, List[str]], 
        return_attention_mask: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Encode text prompts into embeddings.
        
        Args:
            prompts: Single prompt string or list of prompts
            return_attention_mask: Whether to return attention mask
            
        Returns:
            torch.Tensor: Text embeddings of shape (batch_size, embedding_dim)
            or tuple (embeddings, attention_mask) if return_attention_mask=True
        """
        # Handle single string input
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Tokenize
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            
            # Use pooled output (CLS token embedding)
            embeddings = outputs.pooler_output
            
            # Project to desired dimension if needed
            if self.projection is not None:
                embeddings = self.projection(embeddings)
        
        if return_attention_mask:
            return embeddings, tokens.attention_mask
        else:
            return embeddings
    
    def forward(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """Forward pass for compatibility with nn.Module"""
        return self.encode_text(prompts)
    
    def get_null_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get null/empty text embedding for unconditional generation or CFG.
        
        Args:
            batch_size: Number of null embeddings to generate
            
        Returns:
            torch.Tensor: Null embeddings of shape (batch_size, embedding_dim)
        """
        null_prompts = [""] * batch_size
        return self.encode_text(null_prompts)
    
    def interpolate_embeddings(
        self, 
        prompt1: str, 
        prompt2: str, 
        steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two text prompts in embedding space.
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            steps: Number of interpolation steps
            
        Returns:
            torch.Tensor: Interpolated embeddings of shape (steps, embedding_dim)
        """
        emb1 = self.encode_text(prompt1)
        emb2 = self.encode_text(prompt2)
        
        # Linear interpolation
        alphas = torch.linspace(0, 1, steps, device=self.device).unsqueeze(1)
        interpolated = (1 - alphas) * emb1 + alphas * emb2
        
        return interpolated


class TerrainPromptProcessor:
    """
    Preprocessor for terrain-specific prompts to enhance CLIP embeddings.
    """
    
    def __init__(self):
        # Terrain-specific keywords and their enhanced descriptions
        self.terrain_keywords = {
            'mountain': 'tall rocky mountain peaks with steep slopes',
            'mountains': 'tall rocky mountain peaks with steep slopes',
            'hill': 'gentle rolling hills with smooth elevation changes',
            'hills': 'gentle rolling hills with smooth elevation changes',
            'valley': 'deep valley depression between elevated terrain',
            'valleys': 'deep valley depressions between elevated terrain',
            'river': 'meandering water river cutting through landscape',
            'rivers': 'meandering water rivers cutting through landscape',
            'lake': 'calm water lake surrounded by terrain',
            'lakes': 'calm water lakes surrounded by terrain',
            'ocean': 'vast ocean water extending to horizon',
            'sea': 'vast sea water extending to horizon',
            'desert': 'arid sandy desert with dunes and sparse vegetation',
            'forest': 'dense forest with trees and vegetation coverage',
            'jungle': 'thick jungle with dense tropical vegetation',
            'canyon': 'deep rocky canyon with steep cliff walls',
            'cliff': 'vertical cliff face with steep rocky walls',
            'plateau': 'flat elevated plateau terrain with sharp edges',
            'plain': 'flat plains with minimal elevation variation',
            'plains': 'flat plains with minimal elevation variation',
            'volcanic': 'volcanic terrain with crater and lava rock',
            'glacier': 'icy glacier with snow and ice formations',
            'tundra': 'cold tundra with sparse vegetation and permafrost'
        }
        
        # Style modifiers
        self.style_modifiers = [
            'realistic detailed terrain',
            'natural landscape',
            'topographic elevation map',
            'geographic heightfield',
            'digital elevation model'
        ]
    
    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance terrain prompts with more descriptive language for better CLIP encoding.
        
        Args:
            prompt: Original terrain prompt
            
        Returns:
            str: Enhanced prompt with better terrain descriptions
        """
        enhanced = prompt.lower()
        
        # Replace terrain keywords with enhanced descriptions
        for keyword, description in self.terrain_keywords.items():
            if keyword in enhanced:
                enhanced = enhanced.replace(keyword, description)
        
        # Add style modifier
        style_modifier = np.random.choice(self.style_modifiers)
        enhanced = f"{enhanced}, {style_modifier}"
        
        return enhanced


class CLIPTextEncoderWithProcessor(CLIPTextEncoder):
    """
    CLIP encoder with terrain-specific prompt processing.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_processor = TerrainPromptProcessor()
    
    def encode_text_with_enhancement(
        self, 
        prompts: Union[str, List[str]], 
        enhance_prompts: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode text with optional prompt enhancement.
        
        Args:
            prompts: Input prompts
            enhance_prompts: Whether to enhance prompts for better terrain descriptions
            
        Returns:
            torch.Tensor: Text embeddings
        """
        if enhance_prompts:
            if isinstance(prompts, str):
                prompts = self.prompt_processor.enhance_prompt(prompts)
            else:
                prompts = [self.prompt_processor.enhance_prompt(p) for p in prompts]
        
        return self.encode_text(prompts, **kwargs)


def test_clip_encoder():
    """Test function for CLIP encoder"""
    try:
        # Initialize encoder
        encoder = CLIPTextEncoderWithProcessor(embedding_dim=512)
        
        # Test prompts
        test_prompts = [
            "mountainous terrain with rivers",
            "desert landscape with sand dunes",
            "forest covered hills with lakes",
            "volcanic terrain with craters"
        ]
        
        print("Testing CLIP encoder...")
        
        # Test single prompt
        embedding = encoder.encode_text("mountain landscape")
        print(f"Single prompt embedding shape: {embedding.shape}")
        
        # Test batch processing
        embeddings = encoder.encode_text(test_prompts)
        print(f"Batch embeddings shape: {embeddings.shape}")
        
        # Test prompt enhancement
        original = "mountain with river"
        enhanced_embedding = encoder.encode_text_with_enhancement(original, enhance_prompts=True)
        regular_embedding = encoder.encode_text_with_enhancement(original, enhance_prompts=False)
        
        print(f"Enhanced embedding shape: {enhanced_embedding.shape}")
        print(f"Embedding difference (enhanced vs regular): {torch.mean(torch.abs(enhanced_embedding - regular_embedding)):.4f}")
        
        # Test null embedding
        null_emb = encoder.get_null_embedding(batch_size=2)
        print(f"Null embedding shape: {null_emb.shape}")
        
        # Test interpolation
        interp_embs = encoder.interpolate_embeddings("mountain", "desert", steps=5)
        print(f"Interpolated embeddings shape: {interp_embs.shape}")
        
        print("CLIP encoder test passed!")
        
    except Exception as e:
        print(f"CLIP encoder test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_clip_encoder()