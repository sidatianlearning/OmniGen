from OmniGen import OmniGenPipeline


print("loading OnmiGen ...")
pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
print("loading Success")


def inference_onmigen(prompt, input_images, height, width):
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=height//16*16, 
        width=width//16*16,
        guidance_scale=3, #2.5 
        img_guidance_scale=1.6,
        num_inference_steps=35,
        seed=0
    )
    image = images[0]
    image.resize((width, height))
    return image





if __name__ == "__main__":
    prompt="The man and the woman hugged each other tightly. The man is in the <img><|image_1|></img> and the woman is in the <img><|image_2|></img>."
    input_images=["./imgs/test_cases/young_trump.jpeg", "./imgs/test_cases/mckenna.jpg"]
    height = 800
    width = 600
    image = inference_onmigen(prompt, input_images, height, width)
    image.save("test.png")