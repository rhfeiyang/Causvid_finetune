#!/usr/bin/env python3

import torch
import sys
sys.path.append('/home/coder/code/video_sketch/libs/CausVid')

from causvid.models import get_diffusion_wrapper

def debug_tensor_format():
    """Debug script to understand the tensor format handling in WAN models"""
    
    # Create a causal WAN model
    generator = get_diffusion_wrapper("causal_wan")()
    
    print("Generator type:", type(generator))
    print("Model type:", type(generator.model))
    
    # Create sample input data similar to what video_regression.py creates
    batch_size = 1
    num_frames = 5
    channels = 16  # latent channels
    height = 60
    width = 104
    
    # This is the format that comes from video_regression.py
    noisy_input = torch.randn(batch_size, num_frames, channels, height, width)
    conditional_dict = {"prompt_embeds": torch.randn(batch_size, 512, 4096)}
    timestep = torch.tensor([[1000, 1000, 1000, 1000, 1000]])  # [B, F]
    
    print(f"\nInput shapes:")
    print(f"noisy_input: {noisy_input.shape}")
    print(f"prompt_embeds: {conditional_dict['prompt_embeds'].shape}")
    print(f"timestep: {timestep.shape}")
    
    # Try to understand what happens when we call the generator
    try:
        # Monkey patch the model to debug what it receives
        original_forward = generator.model._forward_train
        
        def debug_forward_train(x, t, context, seq_len, clip_fea=None, y=None):
            print(f"\nDEBUG: _forward_train called with:")
            print(f"x type: {type(x)}")
            if isinstance(x, torch.Tensor):
                print(f"x shape: {x.shape}")
            elif isinstance(x, list):
                print(f"x is list with {len(x)} elements")
                for i, tensor in enumerate(x):
                    print(f"  x[{i}] shape: {tensor.shape}")
            else:
                print(f"x is {type(x)}")
                
            print(f"t type: {type(t)}, shape: {t.shape if hasattr(t, 'shape') else 'no shape'}")
            print(f"context type: {type(context)}")
            if isinstance(context, torch.Tensor):
                print(f"context shape: {context.shape}")
            elif isinstance(context, list):
                print(f"context is list with {len(context)} elements")
                for i, tensor in enumerate(context):
                    print(f"  context[{i}] shape: {tensor.shape}")
            else:
                print(f"context is {type(context)}")
                
            print(f"seq_len: {seq_len}")
            
            # Don't actually run the forward to avoid errors
            raise RuntimeError("Debug stop")
        
        generator.model._forward_train = debug_forward_train
        
        # Try to call the generator
        result = generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )
        
    except RuntimeError as e:
        if "Debug stop" in str(e):
            print("\nDebug completed successfully")
        else:
            print(f"\nUnexpected error: {e}")
    except Exception as e:
        print(f"\nError during forward call: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tensor_format() 