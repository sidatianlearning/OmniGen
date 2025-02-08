from OmniGen import OmniGenPipeline
import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image


print("loading InsightFace ...")
app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'genderage'])
app.prepare(ctx_id=0, det_size=(640,640))
print("loading OnmiGen ...")
pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
print("loading Success")


def analysis_face(image_file):
    pil_image = Image.open(image_file).convert("RGB")
    np_image = np.array(pil_image)
    cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    faces = app.get(cv_image)
    if len(faces) == 0:
        face_info = {"has_face": False, "gender": "", "age": -1}
    else:
        faces = sorted(faces, key=lambda x: abs((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), reverse=True)
        face = faces[0]
        face_info = {"has_face": True, "gender": face.sex, "age": face.age}
    return face_info


def info_call(face_info):
    call = ""
    if face_info["has_face"]:
        if face_info["age"] >= 16:
            if face_info["gender"] == "F":
                call = "woman"
            elif face_info["gender"] == "M":
                call = "man"
        else:
            if face_info["gender"] == "F":
                call = "girl"
            elif face_info["gender"] == "M":
                call = "boy"
    return call


def generate_prompt(image_file1, image_file2, template):
    template_list = ["standing", "wedding", "graduation", "pet", "rock", "photograph", "broom", "cheetah", "muscles", "beach", "trophy"]
    assert template in template_list, f"template {template} not in template_list(standing, wedding, graduation, pet, rock, photograph, broom, cheetah, muscles, beach, trophy)"

    info_1 = analysis_face(image_file1)
    info_2 = analysis_face(image_file2)
    call_1 = info_call(info_1)
    call_2 = info_call(info_2)
    if len(call_1) > 0 and len(call_2) > 0 and call_1 == call_2:
        if info_1["age"] > info_2["age"]:
            call_1 = "older " + call_1
            call_2 = "younger " + call_2
        else:
            call_1 = "younger " + call_1
            call_2 = "older " + call_2

    # 合影: A man and a woman are standing together. The woman is in the <img><|image_1|></img> and the man is in the <img><|image_2|></img>.
    if template == "standing":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} and a {} are standing together. The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2, call_1, call_2)

    # 拍婚纱照: The couple stands together for their wedding photos, with the man dressed in a formal three-piece suit and tie, and the woman wearing an elaborate wedding gown with a white veil on her head.
    #           The woman is in the <img><|image_1|></img> and the man is in the <img><|image_2|></img>.
    elif template == "wedding":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            if info_1["gender"] != info_2["gender"]:
                dress = "with the man dressed in a formal three-piece suit and tie, and the woman wearing an elaborate wedding gown with a white veil on her head."
            else:
                dress = "with the {} dressed in a formal three-piece suit and tie, and the {} wearing an elaborate wedding gown with a white veil on her head.".format(call_1, call_2)
            prompt = "The couple stands together for their wedding photos, {}. The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(dress, call_1, call_2)

    # 学士服合影: In the picture, a man and a woman are posing for graduation photos, both dressed in graduation attire.
    #             The woman is wearing a black academic gown, a mortarboard, and holds a diploma in her hand. Similarly, the man is also dressed in a black academic gown and wearing a mortarboard.
    #             The woman is in the <img><|image_1|></img> and the man is in the <img><|image_2|></img>.
    elif template == "graduation":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "In the picture, a {} and a {} are posing for graduation photos, both dressed in graduation attire. ".format(call_1, call_2)
            prompt = prompt + "The {} is wearing a black academic gown, a mortarboard, and holds a diploma in her hand. Similarly, the {} is also dressed in a black academic gown and wearing a mortarboard. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 人宠合影: A person is posing for a photo while holding a pet.The woman is in the <img><|image_1|></img>and the pet is in the <img><|image_2|></img>.
    elif template == "pet":
        if len(call_1) == 0:
            prompt = ""
        else:
            prompt = "A {} is posing for a photo while holding a pet. The {} is in the <img><|image_1|></img> and the pet is in the <img><|image_2|></img>.".format(call_1, call_1)

    # 吉他乐队: a man and a woman are performing rock music on stage. The man is holding a wooden guitar, while the woman plays a sleek, stylish guitar. Behind them is the stage setting.
    #           The woman is in the <img><|image_1|></img> and the man is in the <img><|image_2|></img>.
    elif template == "rock":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} and a {} are performing rock music on stage. The {} is holding a wooden guitar, while the {} plays a sleek, stylish guitar. Behind them is the stage setting. ".format(call_1, call_2, call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 老照片合影: In the photograph from an old, thin photo album, a woman and a man are posing together. They appear jointly in this vintage collection, with the overall tone presenting a warm black-and-white color palette.
    #             The woman is in the <img><|image_1|></img> and the man is in the <img><|image_2|></img>.
    elif template == "photograph":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "In the photograph from an old, thin photo album, a {} and a {} are posing together. They appear jointly in this vintage collection, with the overall tone presenting a warm black-and-white color palette. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 扫帚: A little boy and a little girl are sitting together on a suspended broom. The little boy is in the <img><|image_1|></img> and the little girl is in the <img><|image_2|></img>.
    elif template == "broom":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            if call_1 in ["boy", "girl"]:
                call_1 = "little " + call_1
            if call_2 in ["boy", "girl"]:
                call_2 = "little " + call_2
            prompt = "A {} and a {} are sitting together on a suspended broom. The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2, call_1, call_2)

    # 猎豹: A woman stands with a huge cheetah. The woman is in the <img><|image_1|></img> and the cheetah is in the <img><|image_2|></img>.
    elif template == "cheetah":
        if len(call_1) == 0:
            prompt = ""
        else:
            prompt = "A full-body {} stands on the left and a huge cheetah stands on the right. The {} is in the <img><|image_1|></img> and the cheetah is in the <img><|image_2|></img>.".format(call_1, call_1)

    # 肌肉: A man and a woman are standing together, both wearing sportswear. The man has well-developed muscles, especially the muscles on his arms, which are very large. 
    #       The woman is standing next to the man, and her muscles are also well-developed, with very large muscles on her arms. The man is in the <img><|image_1|></img> and the woman is in the <img><|image_2|></img>.
    elif template == "muscles":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            if info_1["gender"] == "M":
                with_1 = "his"
            else:
                with_1 = "her"
            if info_2["gender"] == "M":
                with_2 = "his"
            else:
                with_2 = "her"
            prompt = "A {} and a {} are standing together, both wearing sportswear. The {} has well-developed muscles, especially the muscles on {} arms, which are very large. ".format(call_1, call_2, call_1, with_1)
            prompt = prompt + "The {} is standing next to the {}, and {} muscles are also well-developed, with very large muscles on {} arms. ".format(call_2, call_1, with_2, with_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 海滩: A man and a woman stood on the beach. The setting sun served as the background to illuminate them. The man embraced the woman intimately and tenderly. The scene was filled with a romantic atmosphere. 
    #       The man is in the <img><|image_1|></img> and the woman is in the <img><|image_2|></img>.
    elif template == "beach":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} and a {} stood on the beach. The setting sun served as the background to illuminate them. The {} embraced the {} intimately and tenderly. The scene was filled with a romantic atmosphere. ".format(call_1, call_2, call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 奖杯: A man in a blue and white racing suit and a woman in a red and white racing suit were standing side by side, holding hands and jointly holding a grand trophy. The man is in the <img><|image_1|></img> and the woman is in the <img><|image_2|></img>.
    elif template == "trophy":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} in a blue and white racing suit and a {} in a red and white racing suit were standing side by side, holding hands and jointly holding a grand trophy. " .format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    return prompt

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
    template = "cheetah"
    input_images = ["./imgs/lw/facefun_muban/baozinvlang3.jpg", "./samples/kitten.jpg"]
    prompt = generate_prompt(input_images[0], input_images[1], template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width)
    image.save("./imgs/lw/cheetah-0208.png")

    # prompt="The man and the woman are standing. The man is in the <img><|image_1|></img> and the woman is in the <img><|image_2|></img>."
    # input_images=["./imgs/test_cases/control.jpg", "./imgs/test_cases/mckenna.jpg"]
    # height = 800
    # width = 600
    # image = inference_onmigen(prompt, input_images, height, width)
    # image.save("test.png")
