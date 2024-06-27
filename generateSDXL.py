###############################################################################
# Load up Stable Diffusion XL and its refiner and generates some images.
# Effectively derived from: 
#       https://huggingface.co/docs/diffusers/using-diffusers/sdxl
###############################################################################

from diffusers import DiffusionPipeline
import torch
import os
import time
import datetime

def main():
    # Load up models
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    
    print("Models loaded.")
    
    # Create output directory
    out_dir = "out_images"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  
        
    # For each image...
    print("Beginning generation...")
    start_time = time.time()      
    image_cnt = 10
    for i in range(image_cnt):
        print("Generating image", i)
        # Make prompt based on count
        cnt = (i+1)
        if cnt > 1:
            prompt = str(cnt) + " astronauts "
        else:
            prompt = "1 astronaut "
        
        prompt += "running in the jungle from a giant dog"      
        
        # Generate and refine image
        image = base(
            prompt=prompt,
            num_inference_steps=40,
            denoising_end=0.8,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=40,
            denoising_start=0.8,
            image=image,
        ).images[0]
        
        # Save image
        image.save(os.path.join(out_dir, "%02d.png" % i))
        
    end_time = time.time()
    print("Done!") 

    elapsed = datetime.timedelta(seconds=(end_time-start_time))    
    print("Training time:", str(elapsed))
        
if __name__ == "__main__":
    main()
    