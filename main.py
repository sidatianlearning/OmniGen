from OmniGen import OmniGenPipeline
import cv2
import insightface
import numpy as np
import os
from insightface.app import FaceAnalysis
from PIL import Image


print("loading InsightFace ...")
app_user = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition', 'genderage'])
app_user.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.04)
app_gen = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition', 'genderage'])
app_gen.prepare(ctx_id=0, det_size=(640,640))
print("loading OnmiGen ...")
pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
print("loading Success")


def analysis_face(image_file):
    if len(image_file) == 0:
        face_info = {"has_face": False, "gender": "", "age": -1, "embedding": None}
    elif not os.path.isfile(image_file):
        face_info = {"has_face": False, "gender": "", "age": -1, "embedding": None}
    else:
        pil_image = Image.open(image_file).convert("RGB")
        np_image = np.array(pil_image)
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        faces = app_user.get(cv_image)
        if len(faces) == 0:
            face_info = {"has_face": False, "gender": "", "age": -1, "embedding": None}
        else:
            faces = sorted(faces, key=lambda x: abs((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), reverse=True)
            face = faces[0]
            face_info = {"has_face": True, "gender": face.sex, "age": face.age, "embedding": face.embedding}
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

def get_face_similarity(face, infos):
    def compute_emb_similarity(embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            similarity = -1
        else:
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
    max_similarity = -1
    for info in infos:
        similarity = compute_emb_similarity(face.embedding, info["embedding"])
        if similarity > max_similarity:
            max_similarity = similarity
    return max_similarity

def generate_prompt(image_file1, image_file2, template):
    template_list = [
        "standing", "wedding", "graduation", "pet", "rock", "photograph", "broom", "cheetah", "muscles", "beach", "trophy", "egyptian", "balloons",
        "titanic", "redcar", "brazil", "wallstreet", "convertible", "doctor", "rain", "horse", "kangaroo", "eiffeltower", "wall", "pregnant", "seabed", "mother", "gift", "microphone",
        "elephant", "cartoon", "lizard", "snowflakes", "fireworks", "aurora", "cornfield", "stable", "football", "toolroom", "openairmarket", "rainforest", "spiderman", "aladdin"]
    assert template in template_list, (
        f"template {template} not in template_list(standing, wedding, graduation, pet, rock, photograph, broom, cheetah, muscles, beach, trophy, egyptian, balloons,"
        f"titanic, redcar, brazil, wallstreet, convertible, doctor, rain, horse, kangaroo, eiffeltower, wall, pregnant, seabed, mother, gift, microphone, elephant, "
        f"cartoon, lizard, snowflakes, fireworks, aurora, cornfield, stable, football, toolroom, openairmarket, rainforest, spiderman, aladdin)"
    )

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
            prompt = "A {} stands on the left and a {} stands on the right. The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2, call_1, call_2)

    # 拍婚纱照: The couple stands together for their wedding photos, with the man dressed in a formal three-piece suit and tie, and the woman wearing an elaborate wedding gown with a white veil on her head.
    #           The woman is in the <img><|image_1|></img> and the man is in the <img><|image_2|></img>.
    elif template == "wedding":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            if info_1["gender"] != info_2["gender"]:
                dress = "with the man dressed in a formal three-piece suit and tie, and the woman wearing an elaborate wedding gown with a white veil on her head"
            else:
                dress = "with the {} dressed in a formal three-piece suit and tie, and the {} wearing an elaborate wedding gown with a white veil on her head".format(call_1, call_2)
            prompt = "The {} stands on the left and the {} stands on the right for their wedding photos, {}. The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2, dress, call_1, call_2)

    # 学士服合影: In the picture, a man and a woman are posing for graduation photos, both dressed in graduation attire.
    #             The woman is wearing a black academic gown, a mortarboard, and holds a diploma in her hand. Similarly, the man is also dressed in a black academic gown and wearing a mortarboard.
    #             The woman is in the <img><|image_1|></img> and the man is in the <img><|image_2|></img>.
    elif template == "graduation":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "In the picture, a {} on the left and a {} on the right are posing for graduation photos, both dressed in graduation attire. ".format(call_1, call_2)
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
            prompt = "A {} on the left and a {} on the right are performing rock music on stage. The {} is holding a wooden guitar, while the {} plays a sleek, stylish guitar. Behind them is the stage setting. ".format(call_1, call_2, call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 老照片合影: In the photograph from an old, thin photo album, a woman and a man are posing together. They appear jointly in this vintage collection, with the overall tone presenting a warm black-and-white color palette.
    #             The woman is in the <img><|image_1|></img> and the man is in the <img><|image_2|></img>.
    elif template == "photograph":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "In the photograph from an old, thin photo album, a {} on the left and a {} on the right are posing together. They appear jointly in this vintage collection, with the overall tone presenting a warm black-and-white color palette. ".format(call_1, call_2)
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
            prompt = "A {} stands on the left and a {} stands on the right, both wearing sportswear. The {} has well-developed muscles, especially the muscles on {} arms, which are very large. ".format(call_1, call_2, call_1, with_1)
            prompt = prompt + "The {} is standing next to the {}, and {} muscles are also well-developed, with very large muscles on {} arms. ".format(call_2, call_1, with_2, with_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 海滩: A man and a woman stood on the beach. The setting sun served as the background to illuminate them. The man embraced the woman intimately and tenderly. The scene was filled with a romantic atmosphere. 
    #       The man is in the <img><|image_1|></img> and the woman is in the <img><|image_2|></img>.
    elif template == "beach":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "On the beach, A {} stood on the left and a {} stood on the right. The setting sun served as the background to illuminate them. The {} embraced the {} intimately and tenderly. The scene was filled with a romantic atmosphere. ".format(call_1, call_2, call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 奖杯: A man in a blue and white racing suit and a woman in a red and white racing suit were standing side by side, holding hands and jointly holding a grand trophy. The man is in the <img><|image_1|></img> and the woman is in the <img><|image_2|></img>.
    elif template == "trophy":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stood on the left in a blue and white racing suit and a {} stood on the right in a red and white racing suit, holding hands and jointly holding a grand trophy. " .format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 埃及：A man and a woman stand in front of an ancient Egyptian temple, bathed in the golden sunlight.
    #       The man is tall and strong, with long golden brown hair tied into a low ponytail, wearing a white knee-length skirt, a wide gold belt around his waist, and gold ornaments on his arms, with a majestic temperament.
    #       The woman has black hair that reaches her shoulders, and her eyeliner outlines her mysterious charm. She wears a close-fitting white dress and a gold collar with sapphires around her neck, just like an ancient Egyptian queen. 
    #       They stand side by side, with deep eyes, and the background is the majestic temple and pyramids. The setting sun casts afterglow, making the whole picture full of mystery and solemnity.
    elif template == "egyptian":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, in front of an ancient Egyptian temple, bathed in the golden sunlight. ".format(call_1, call_2)
            if info_1["gender"] != info_2["gender"]:
                prompt = prompt + "The man is tall and strong, with long golden brown hair tied into a low ponytail, wearing a white knee-length skirt, a wide gold belt around his waist, and gold ornaments on his arms, with a majestic temperament. "
                prompt = prompt + "The woman has black hair that reaches her shoulders, and her eyeliner outlines her mysterious charm. She wears a close-fitting white dress and a gold collar with sapphires around her neck, just like an ancient Egyptian queen. "
            else:
                if info_1["gender"] == "M":
                    with_ = "his"
                else:
                    with_ = "her"
                prompt = prompt + "The {} is tall and strong, with long golden brown hair tied into a low ponytail, wearing a white knee-length skirt, a wide gold belt around {} waist, and gold ornaments on {} arms, with a majestic temperament. ".format(call_1, with_, with_)
                prompt = prompt + "The {} has black hair that reaches {} shoulders, and {} eyeliner outlines {} mysterious charm. She wears a close-fitting white dress and a gold collar with sapphires around {} neck, just like an ancient Egyptian queen. ".format(call_2, with_, with_, with_, with_)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 气球：In this painting, a couple is exuding a sweet affection. The man is embracing the woman in his strong arms. They are smiling towards the camera, and around them are a large bouquet of heart-shaped balloons.
    #       The background features an outdoor wooden walkway and the Eiffel Tower. The entire scene exudes a romantic and warm atmosphere.
    elif template == "balloons":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "In this painting, a couple is exuding a sweet affection. "
            if info_1["gender"] != info_2["gender"]:
                prompt = prompt + "The man is embracing the woman in his strong arms. "
            else:
                if info_1["gender"] == "M":
                    with_ = "his"
                else:
                    with_ = "her"
                prompt = prompt + "The {} is embracing the {} in {} strong arms. ".format(call_1, call_2, with_)
            prompt = prompt + "They are smiling towards the camera, and around them are a large bouquet of heart-shaped balloons. The background features an outdoor wooden walkway and the Eiffel Tower. The entire scene exudes a romantic and warm atmosphere. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 泰坦：A couple stood at the bow of the ship, imitating the classic flying posture of the Titanic. The woman was in the front, her arms spread wide, her head slightly raised, her eyes closed, feeling the sea breeze blowing towards her, with a relaxed and excited smile on her lips.
    #       Her long hair was fluttering in the wind, and her skirt was gently lifted. The man stood behind her, gently wrapping his arms around her waist, with a smile on his lips. 
    #       Behind them, the waves were surging and crashing, the setting sun tinted the sky with warm golden red, the ship was moving slowly, and the sea breeze blew over the two of them, carrying the scent of the sea.
    elif template == "titanic":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A couple stood at the bow of the ship, imitating the classic flying posture of the Titanic. "
            if info_1["gender"] != info_2["gender"]:
                prompt = prompt + "The woman was in the front, her arms spread wide, her head slightly raised, her eyes closed, feeling the sea breeze blowing towards her, with a relaxed and excited smile on her lips. Her long hair was fluttering in the wind, and her skirt was gently lifted. "
                prompt = prompt + "The man stood behind her, gently wrapping his arms around her waist, with a smile on his lips. "
            else:
                if info_1["gender"] == "M":
                    with_ = "his"
                else:
                    with_ = "her"
                prompt = prompt + "The {} was in the front, {} arms spread wide, {} head slightly raised, {} eyes closed, feeling the sea breeze blowing towards her, with a relaxed and excited smile on {} lips. {} long hair was fluttering in the wind, and {} skirt was gently lifted. ".format(call_1, with_, with_, with_, with_, with_, with_)
                prompt = prompt + "The {} stood behind her, gently wrapping {} arms around {} waist, with a smile on {} lips. ".format(call_2, with_, with_, with_)
            prompt = prompt + "Behind them, the waves were surging and crashing, the setting sun tinted the sky with warm golden red, the ship was moving slowly, and the sea breeze blew over the two of them, carrying the scent of the sea."
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 红色跑车：The man and the woman stood together, smiling towards the camera. The background was a serene outdoor scene, with a vintage red sports car behind them, neatly trimmed green plants and a white fence. 
    #           The sunlight poured in, creating a warm and romantic atmosphere. The overall picture exuded the charm of retro style and the sweet atmosphere of love.
    elif template == "redcar":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "The {} stood on the left and the {} stood on the right, smiling towards the camera. The background was a serene outdoor scene, with a vintage red sports car behind them, neatly trimmed green plants and a white fence. ".format(call_1, call_2)
            prompt = prompt + "The sunlight poured in, creating a warm and romantic atmosphere. The overall picture exuded the charm of retro style and the sweet atmosphere of love. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 巴西狂欢节：The woman is wearing a large yellow feather headdress with colorful parts, which makes her look very eye-catching and she has a bright smile. 
    #               She is wearing a tight blue and yellow outfit with a lot of feathers and sequins on the chest and cuffs, and the skirt below the waist is also rich in color and decorative details. 
    #               The man is wearing a colorful and distinctive outfit. He wears a large red feather headdress with gold decorations. 
    #               The shoulders and arms are decorated with a lot of red and orange feathers, forming an exaggerated shape. He is naked from the waist, showing his strong muscles, and his lower body is decorated with red and gold clothes, also decorated with feathers. 
    #               The overall outfit may be used for carnival or similar festivals. In the background, a street and some people can be seen, which is full of joyful atmosphere. This kind of dress is common in Latin American carnival and other festivals.
    elif template == "brazil":
        if len(call_1) == 0 or len(call_2) == 0 or info_1["gender"] == info_2["gender"]:
            prompt = ""
        else:
            prompt = "The woman is wearing a large yellow feather headdress with colorful parts, which makes her look very eye-catching and she has a bright smile. "
            prompt = prompt + "She is wearing a tight blue and yellow outfit with a lot of feathers and sequins on the chest and cuffs, and the skirt below the waist is also rich in color and decorative details. "
            prompt = prompt + "The man is wearing a colorful and distinctive outfit. He wears a large red feather headdress with gold decorations. "
            prompt = prompt + "The shoulders and arms are decorated with a lot of red and orange feathers, forming an exaggerated shape. He is naked from the waist, showing his strong muscles, and his lower body is decorated with red and gold clothes, also decorated with feathers. "
            prompt = prompt + "The overall outfit may be used for carnival or similar festivals. In the background, a street and some people can be seen, which is full of joyful atmosphere. This kind of dress is common in Latin American carnival and other festivals. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 华尔街：The woman and the man stood together, behind them was a huge floor-to-ceiling window, and outside the window were skyscrapers with neon lights twinkling.
    elif template == "wallstreet":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "The {} stood on the left and the {} stood on the right, behind them was a huge floor-to-ceiling window, and outside the window were skyscrapers with neon lights twinkling. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 敞篷车：A man and a woman were sitting in a red sports car.
    elif template == "convertible":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} and a {} were sitting in a red sports car. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 医生：A man and a woman, both wearing white coats and with a stethoscope around their necks, stood inside a hospital.
    elif template == "doctor":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stood on the left and a {} stood on the right, both wearing white coats and with a stethoscope around their necks, stood inside a hospital. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 雨中：Men and women stood on a street, the rain pouring down on them, their clothes soaked.
    elif template == "rain":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "On a street, the {} stands on the left and the {} stands on the right, the rain pouring down on them, their clothes soaked. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 骑马：A woman is riding a horse.
    elif template == "horse":
        if len(call_1) == 0:
            prompt = ""
        else:
            prompt = "A {} is riding a horse. The {} is in the <img><|image_1|></img> and the horse is in the <img><|image_2|></img>.".format(call_1, call_1)

    # 袋鼠：A man wearing boxing gloves was standing on the boxing ring. Next to him was a kangaroo. The kangaroo had very well-developed muscles. Many people on the audience stand behind were waiting for this exciting match.
    elif template == "kangaroo":
        if len(call_1) == 0:
            prompt = ""
        else:
            if info_1["gender"] == "M":
                with_1 = "him"
            else:
                with_1 = "her"
            prompt = "A {} wearing boxing gloves was standing on the boxing ring. Next to {} was a kangaroo. The kangaroo had very well-developed muscles. ".format(call_1, with_1)
            prompt = prompt + "Many people on the audience stand behind were waiting for this exciting match. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the kangaroo is in the <img><|image_2|></img>.".format(call_1)

    # 埃菲尔铁塔：A man and a woman stood together under the Eiffel Tower and the cherry blossom trees.
    elif template == "eiffeltower":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, under the Eiffel Tower and the cherry blossom trees. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 壁咚：The man and the woman are standing together. There is a wall beside the woman.
    elif template == "wall":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "The {} stands on the left and the {} stands on the right. ".format(call_1, call_2)
            if info_1["gender"] == "M" and info_2["gender"] == "M":
                prompt = prompt + "There is a wall beside the {}. " .format(call_1)
            else:
                prompt = prompt + "There is a wall beside the woman. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 怀孕：The woman is pregnant and the man is standing beside her.
    elif template == "pregnant":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        elif info_1["gender"] == info_2["gender"]:
            prompt = ""
        else:
            prompt = "The pregnant woman stands on the right, and the man stands on the left. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 海底：This is a photo taken from the seabed. A couple of people are in the blue underwater world. The man is gently embracing the woman.
    #       Around them, some beautiful dolphins are swimming gracefully, and various kinds of small fish are darting and swimming among the coral reefs.
    #       The bubbles rising from the bottom of the water add a sense of movement to the picture.
    elif template == "seabed":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "This is a photo taken from the seabed. A couple of people are in the blue underwater world. The {} stands on the left and the {} stands on the right.".format(call_1, call_2)
            if info_1["gender"] == info_2["gender"]:
                prompt = prompt + "The {} is gently embracing the {}. ".format(call_1, call_2)
            else:
                prompt = prompt + "The man is gently embracing the woman. "
            prompt = prompt + "Around them, some beautiful dolphins are swimming gracefully, and various kinds of small fish are darting and swimming among the coral reefs. "
            prompt = prompt + "The bubbles rising from the bottom of the water add a sense of movement to the picture. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 母亲：Two women stand together, with the background being the blue sky and white clouds outside.
    elif template == "mother":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, with the background being the blue sky and white clouds outside. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 礼物：A man and a woman stand together.
    elif template == "gift":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 麦克风：A woman was standing on a podium, with a microphone in front of her.
    elif template == "microphone":
        if len(call_1) == 0:
            prompt = ""
        else:
            prompt = "A {} was standing on a podium, with a microphone in front of her. ".format(call_1)
            prompt = prompt + "The {} is in the <img><|image_1|></img>.".format(call_1)

    # 大象：A person riding an elephant。
    elif template == "elephant":
        if len(call_1) == 0:
            prompt = ""
        else:
            prompt = "A {} riding an elephant. ".format(call_1)
            prompt = prompt + "The {} is in the <img><|image_1|></img>.".format(call_1)

    # 卡通：This person has been transformed into a cartoon style.
    elif template == "cartoon":
        if len(call_1) == 0:
            prompt = ""
        else:
            prompt = "This {} has been transformed into a cartoon style. ".format(call_1)
            prompt = prompt + "The {} is in the <img><|image_1|></img>.".format(call_1)

    # 蜥蜴：This person is standing on the African savannah, holding a huge and brightly-colored lizard in his arms.
    elif template == "lizard":
        if len(call_1) == 0:
            prompt = ""
        else:
            prompt = "This {} is standing on the African savannah, holding a huge and brightly-colored lizard in his arms. ".format(call_1)
            prompt = prompt + "The {} is in the <img><|image_1|></img>.".format(call_1)

    # 雪中kiss：A man and a woman stand together, with snowflakes falling behind them.
    elif template == "snowflakes":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, with snowflakes falling behind them. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)
    
    # 烟花亲吻：A man and a woman stand together, with the characters in the center of the picture. 
    #           The sky behind them presents a colorful fireworks show. 
    #           In the night sky, golden, white, pink, and orange fireworks bloom in various shapes: some are as bright as blooming flowers, some are like trailing meteors, and some are scattered into dots of light. 
    #           The light of the fireworks is particularly dazzling against the dark night sky, and the smoke in the air adds a hazy beauty.
    elif template == "fireworks":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, with the characters in the center of the picture. ".format(call_1, call_2)
            prompt = prompt + "The sky behind them presents a colorful fireworks show. "
            prompt = prompt + "In the night sky, golden, white, pink, and orange fireworks bloom in various shapes: some are as bright as blooming flowers, some are like trailing meteors, and some are scattered into dots of light. "
            prompt = prompt + "The light of the fireworks is particularly dazzling against the dark night sky, and the smoke in the air adds a hazy beauty. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 极光亲吻：A man and a woman stand together, with beautiful colorful aurora and snow-capped mountains behind them.
    elif template == "aurora":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, with beautiful colorful aurora and snow-capped mountains behind them. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 玉米地：A man and a woman stand together, with a cornfield in the background. In the background, the corn plants are tall, some of their leaves have turned yellow, and dry corn stalks are scattered on the ground.
    elif template == "cornfield":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, with a cornfield in the background. ".format(call_1, call_2)
            prompt = prompt + "In the background, the corn plants are tall, some of their leaves have turned yellow, and dry corn stalks are scattered on the ground. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 马棚：A man and a woman are standing together, with an American stable in the background. With a wooden frame, there is a brown horse behind the two people.
    elif template == "stable":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, with an American stable in the background. ".format(call_1, call_2)
            prompt = prompt + "With a wooden frame, there is a brown horse behind the two people. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 橄榄球：A man and a woman stand together, with a large open-air stadium in the background. Under the blue sky and white clouds, the spectator stands surround the green American football field layer upon layer.
    elif template == "football":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right, with a large open-air stadium in the background. ".format(call_1, call_2)
            prompt = prompt + "Under the blue sky and white clouds, the spectator stands surround the green American football field layer upon layer. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 工具房：A man and a woman are standing together. The background is a tool room. Tools such as pliers, wrenches and saws are hung on the wall, and there is a toolbox behind them.
    elif template == "toolroom":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stands on the left and a {} stands on the right. ".format(call_1, call_2)
            prompt = prompt + "The background is a tool room. Tools such as pliers, wrenches and saws are hung on the wall, and there is a toolbox behind them. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 露天市场：A man and a woman stand side by side in an open-air market in Africa. The sun shines through the gaps in the green sunshade above their heads, casting mottled light and shadows on them. 
    #           Surrounding them are noisy streets, stalls lined with colorful clothes, and cartons, plastic bags and cargo packages scattered on the ground. 
    #           The two stand in the middle of a slightly open alley, not far away from them is a motorcycle and several cargo boxes waiting to be moved. 
    #           In the background are pedestrians shuttling back and forth, vendors hawking and dogs wandering around casually. 
    #           The whole picture is rich in color and warm in atmosphere, showing the unique chaotic vitality and life atmosphere of the African market.
    elif template == "openairmarket":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} and a {} stand side by side in an open-air market in Africa. ".format(call_1, call_2)
            prompt = prompt + "The sun shines through the gaps in the green sunshade above their heads, casting mottled light and shadows on them. "
            prompt = prompt + "Surrounding them are noisy streets, stalls lined with colorful clothes, and cartons, plastic bags and cargo packages scattered on the ground. "
            prompt = prompt + "The two stand in the middle of a slightly open alley, not far away from them is a motorcycle and several cargo boxes waiting to be moved. "
            prompt = prompt + "In the background are pedestrians shuttling back and forth, vendors hawking and dogs wandering around casually. "
            prompt = prompt + "The whole picture is rich in color and warm in atmosphere, showing the unique chaotic vitality and life atmosphere of the African market. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)
    
    # 下雨雨林：A man and a woman were standing together. It was raining heavily. The background was a thick tropical rainforest, surrounded by tall green plants with luxuriant leaves and various shapes. 
    #           There was a babbling brook on the ground and a small waterfall could be seen in the distance.
    elif template == "rainforest":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} stood on the left and a {} stood on the right. ".format(call_1, call_2)
            prompt = prompt + "It was raining heavily. The background was a thick tropical rainforest, surrounded by tall green plants with luxuriant leaves and various shapes. "
            prompt = prompt + "There was a babbling brook on the ground and a small waterfall could be seen in the distance. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 蜘蛛侠：A man, wearing a red and black mesh bodysuit, leaps onto the ground, holding hands with a woman and a spider web with the other. 
    #           The woman stands next to him. The background is a vibrant, cinematic backdrop of New York skyscrapers.
    elif template == "spiderman":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        elif info_1["gender"] == info_2["gender"]:
            prompt = ""
        else:
            prompt = "A man, wearing a red and black mesh bodysuit, leaps onto the ground, holding hands with a woman and a spider web with the other. "
            prompt = prompt + "The woman stands next to him. The background is a vibrant, cinematic backdrop of New York skyscrapers. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)
    
    # 阿拉丁：A man and a woman are sitting naturally on a gleaming magic carpet and flying in the air. 
    #       The man is wearing a white shirt with red decorations, paired with a deep purple sleeveless vest, and white pants. The overall style has the characteristics of traditional Arab clothing. 
    #       The woman is wearing a blue-green plunging top, paired with the same color pants, and is adorned with gold necklaces, earrings, etc. 
    #       The clothing has bright colors. The background is a starry night sky and the distant city lights, with a full moon in the sky. The scene is filled with a romantic atmosphere.
    elif template == "aladdin":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        elif info_1["gender"] == info_2["gender"]:
            prompt = ""
        else:
            prompt = "A man and a woman are sitting naturally on a gleaming magic carpet and flying in the air. "
            prompt = prompt + "The man is wearing a white shirt with red decorations, paired with a deep purple sleeveless vest, and white pants. The overall style has the characteristics of traditional Arab clothing. "
            prompt = prompt + "The woman is wearing a blue-green plunging top, paired with the same color pants, and is adorned with gold necklaces, earrings, etc. The clothing has bright colors. "
            prompt = prompt + "The background is a starry night sky and the distant city lights, with a full moon in the sky. The scene is filled with a romantic atmosphere. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)
    return prompt

def generate_prompt_for_3(image_file1, image_file2, image_file3, template):
    template_list = ["photo"]
    assert template in template_list, f"template {template} not in template_list(photo)"

    info_1 = analysis_face(image_file1)
    info_2 = analysis_face(image_file2)
    info_3 = analysis_face(image_file3)
    call_1 = info_call(info_1)
    call_2 = info_call(info_2)
    call_3 = info_call(info_3)

    if template == "photo":
        if len(call_1) == 0 or len(call_2) == 0 or len(call_3) == 0:
            prompt = ""
        elif info_1["gender"] != "M" or info_2["gender"] != "F" or info_3["gender"] != "F":
            prompt = ""
        else:
            prompt = "Three people are posing for a group photo. The man is standing in the middle, holding the two women in his arms. "
            prompt = prompt + "The man is in the <img><|image_1|></img>, the left woman is in the <img><|image_2|></img>, and the right woman is in the <img><|image_3|></img>."
    return prompt



def generate_prompts_styleme(image_file, template):
    template_list = ["identification", "workplace", "exotic", "thanksgiving-lady", "thanksgiving-baby"]
    assert template in template_list, f"template {template} not in template_list (identification, workplace, exotic, thanksgiving-lady, thanksgiving-baby)"

    info = analysis_face(image_file)
    call = info_call(info)
    if info["gender"] == "M":
        pronoun = "He"
        possessive = "his"
        with_ = "him"
    else:
        pronoun = "She"
        possessive = "her"
        with_ = "her"

    prompts = []
    if len(call) == 0:
        return prompts

    # 证件照
    # A person is wearing a formal white shirt and a red checked tie. He is smiling at the camera with his hands naturally hanging down. The background color is dark blue.
    # A person is wearing a formal set of dark purple suit and a white polka-dot tie, smiling at the camera. The background color is dark gray.
    # A person is wearing a short-sleeved shirt and a blue tie, smiling at the camera. The background is in a light grey tone.
    # A person is wearing a formal black suit and a blue tie, smiling at the camera. The background is in a dark grey tone.
    if template == "identification":
        prompt = f"A {call} is wearing a formal white shirt and a red checked tie. {pronoun} is smiling at the camera with his hands naturally hanging down. The background color is dark blue. The {call} is in the <img><|image_1|></img>."
        prompts.append(prompt)
        prompt = f"A {call} is wearing a formal set of dark purple suit and a white polka-dot tie, smiling at the camera. The background color is dark gray. The {call} is in the <img><|image_1|></img>."
        prompts.append(prompt)
        prompt = f"A {call} is wearing a short-sleeved shirt and a blue tie, smiling at the camera. The background is in a light grey tone. The {call} is in the <img><|image_1|></img>."
        prompts.append(prompt)
        prompt = f"A {call} is wearing a formal black suit and a blue tie, smiling at the camera. The background is in a dark grey tone. The {call} is in the <img><|image_1|></img>."
        prompts.append(prompt)

    # 职场照
    # A person is dressed in a dark grey suit, paired with a white shirt. Standing in front of the glass window, the background shows the tall buildings of the city. He slightly leans to the side, with a confident and concentrated expression, and his hands are crossed over his chest.
    # A person is sitting at the conference table, dressed in a black business suit and wearing a simple white shirt. He is holding a document in his hand and his gaze is firm and directed forward. The background is a bright modern conference room.
    # A person is standing with his arms crossed in a dark blue double-breasted suit, wearing a black shirt underneath. The background is a wooden bookshelf with books and green plants placed on it.
    # A person is sitting on a black office chair in a light khaki suit. One hand is placed on a notebook, and a laptop is placed in front of him/her. The background is blue blinds.
    elif template == "workplace":
        prompt = (
            f"A {call} is dressed in a dark grey suit, paired with a white shirt. Standing in front of the glass window, the background shows the tall buildings of the city. "
            f"{pronoun} slightly leans to the side, with a confident and concentrated expression, and {possessive} hands are crossed over {possessive} chest. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"A {call} is sitting at the conference table, dressed in a black business suit and wearing a simple white shirt. "
            f"{pronoun} is holding a document in {possessive} hand and {possessive} gaze is firm and directed forward. The background is a bright modern conference room. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"A {call} is standing with {possessive} arms crossed in a dark blue double-breasted suit, wearing a black shirt underneath. The background is a wooden bookshelf with books and green plants placed on it. "
            f"The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"A {call} is sitting on a black office chair in a light khaki suit. One hand is placed on a notebook, and a laptop is placed in front of {with_}. The background is blue blinds."
            f"The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)

    # 异域风情
    # 女1：The woman is dressed in a light brown robe, with a belt featuring golden round buckles around her waist. She wears a delicate gold necklace around her neck, brown wristbands wrapped around her wrists, and a hood of the same color on her head. 
    #       Beside her is a large camel, and in the background is a desert, with two stone statues of a similar ancient Egyptian style behind her.
    # 女2：The woman wore a magnificent headband on her head and her hair was braided. She was dressed in golden attire with intricate patterns and decorations on it, complemented by a wide golden belt and multiple layers of necklaces. 
    #       She wore arm rings on her arms. She was sitting on a chair, her eyes deep-set, presenting an overall image of nobility and mystery. Beside her was a white-gold tiger, and the background was magnificent.
    # 女3：A woman is gracefully standing by the seaside, dressed in a short, exotic jacket with intricate golden embroidery along the collar, adding a touch of regal charm to her look. 
    #       The jacket’s design is simple yet rich with cultural details, exuding an air of luxury and refinement. Her lower body is adorned with a flowing, beige skirt that moves with the breeze, and a wide belt decorated with unique patterns that cinches at her waist, accentuating her figure. 
    #       In her hand, she holds a majestic white horse, its pure coat glowing against the backdrop of the ocean. The scene captures her elegance and strength, with her attire reflecting a blend of exotic beauty and grace.
    # 女4：The woman wore large golden earrings and a thick necklace. Her body was glittering with golden sequins. She leaned against a black panther. The panther had sharp eyes and its yellow eyes were very striking. The background was water surface.
    # 男1：The man is wearing a light brown robe with a belt with a round gold buckle around his waist. The design of the robe is simple yet elegant. He wears an exquisite gold necklace around his neck and a few brown leather bracelets around his wrists, adding a sense of strength. 
    #       He wears a hood of the same color as his robe on his head, which droops gently, revealing a pair of deep eyes and a resolute face. He stands next to a tall camel, whose hair is a warm brown, giving him a calm temperament. 
    #       The background is a vast desert with some hard stones scattered on the ground, and in the distance there are two ancient stone statues, which seem to be derived from the ancient Egyptian style, mysterious and solemn. 
    #       The whole scene creates a mysterious and magnificent atmosphere. The man's posture is steady and calm, exuding a unique charm that blends with nature and history.
    # 男2：The man wore a gorgeous crown, and his hair was carefully braided into fine braids, showing a kingly majesty. He was dressed in a golden suit, with intricate and exquisite patterns and decorations carved on it. 
    #       The golden fabric shone with a luxurious luster under the light, and a wide golden belt was tied around his waist, adding a sense of solemnity. He wore several layers of gorgeous gold necklaces on his neck, each of which was unique and emitted a faint metallic glow, highlighting his noble status. 
    #       He wore heavy arm rings on his arms, with a simple and exquisite design, as if carrying a long history and power.
    # 男3：The man stands by the sea, dressed in an exotic, short jacket adorned with intricate golden embroidery along the collar. The jacket’s simple cut contrasts with the ornate detailing, creating a unique blend of sophistication and understated luxury. 
    #       The golden collar adds a regal touch, elevating the overall look. He wears flowing beige trousers, their hems slightly grazing the ground, and a wide, intricately patterned belt that cinches at his waist. 
    #       The belt’s design is both simple and layered, enhancing his figure while maintaining an elegant balance.
    # 男4：The man is wearing a dark robe with exquisite patterns and intricate designs. The colors are deep and layered, showing a noble and mysterious atmosphere. The collar is low, revealing a delicate collarbone, and the hem of the robe is wide and elegantly draped to the ground. 
    #       A gorgeous belt with exquisite decorations is tied around his waist, highlighting his elegance and strength. The cuffs are wide and decorated with exquisite embroidery patterns, which flutter gently in the breeze, adding a sense of movement.
    elif template == "exotic":
        if info["gender"] == "M":
            prompt = (
                f"The man is wearing a light brown robe with a belt with a round gold buckle around his waist. "
                f"The design of the robe is simple yet elegant. He wears an exquisite gold necklace around his neck and a few brown leather bracelets around his wrists, adding a sense of strength. "
                f"He wears a hood of the same color as his robe on his head, which droops gently, revealing a pair of deep eyes and a resolute face. He stands next to a tall camel, whose hair is a warm brown, giving him a calm temperament. "
                f"The background is a vast desert with some hard stones scattered on the ground, and in the distance there are two ancient stone statues, which seem to be derived from the ancient Egyptian style, mysterious and solemn. "
                f"The whole scene creates a mysterious and magnificent atmosphere. The man's posture is steady and calm, exuding a unique charm that blends with nature and history. "
                f"The man is in the <img><|image_1|></img>."
            )
            prompts.append(prompt)
            prompt = (
                f"The man wore a gorgeous crown, and his hair was carefully braided into fine braids, showing a kingly majesty. He was dressed in a golden suit, with intricate and exquisite patterns and decorations carved on it. "
                f"The golden fabric shone with a luxurious luster under the light, and a wide golden belt was tied around his waist, adding a sense of solemnity. "
                f"He wore several layers of gorgeous gold necklaces on his neck, each of which was unique and emitted a faint metallic glow, highlighting his noble status. "
                f"He wore heavy arm rings on his arms, with a simple and exquisite design, as if carrying a long history and power. "
                f"The man is in the <img><|image_1|></img>."
            )
            prompts.append(prompt)
            prompt = (
                f"The man stands by the sea, dressed in an exotic, short jacket adorned with intricate golden embroidery along the collar. The jacket's simple cut contrasts with the ornate detailing, creating a unique blend of sophistication and understated luxury. "
                f"The golden collar adds a regal touch, elevating the overall look. He wears flowing beige trousers, their hems slightly grazing the ground, and a wide, intricately patterned belt that cinches at his waist. "
                f"The belt's design is both simple and layered, enhancing his figure while maintaining an elegant balance. "
                f"The man is in the <img><|image_1|></img>."
            )
            prompts.append(prompt)
            prompt = (
                f"The man is wearing a dark robe with exquisite patterns and intricate designs. The colors are deep and layered, showing a noble and mysterious atmosphere. The collar is low, revealing a delicate collarbone, and the hem of the robe is wide and elegantly draped to the ground. "
                f"A gorgeous belt with exquisite decorations is tied around his waist, highlighting his elegance and strength. The cuffs are wide and decorated with exquisite embroidery patterns, which flutter gently in the breeze, adding a sense of movement. "
                f"The man is in the <img><|image_1|></img>."
            )
            prompts.append(prompt)
        else:
            prompt = (
                f"The woman is dressed in a light brown robe, with a belt featuring golden round buckles around her waist. She wears a delicate gold necklace around her neck, brown wristbands wrapped around her wrists, and a hood of the same color on her head. "
                f"Beside her is a large camel, and in the background is a desert, with two stone statues of a similar ancient Egyptian style behind her. "
                f"The woman is in the <img><|image_1|></img>."
            )
            prompts.append(prompt)
            prompt = (
                f"The woman wore a magnificent headband on her head and her hair was braided. She was dressed in golden attire with intricate patterns and decorations on it, complemented by a wide golden belt and multiple layers of necklaces. "
                f"She wore arm rings on her arms. She was sitting on a chair, her eyes deep-set, presenting an overall image of nobility and mystery. Beside her was a white-gold tiger, and the background was magnificent. "
                f"The woman is in the <img><|image_1|></img>."
            )
            prompts.append(prompt)
            prompt = (
                f"A woman is gracefully standing by the seaside, dressed in a short, exotic jacket with intricate golden embroidery along the collar, adding a touch of regal charm to her look. "
                f"The jacket's design is simple yet rich with cultural details, exuding an air of luxury and refinement. Her lower body is adorned with a flowing, beige skirt that moves with the breeze, and a wide belt decorated with unique patterns that cinches at her waist, accentuating her figure. "
                f"In her hand, she holds a majestic white horse, its pure coat glowing against the backdrop of the ocean. The scene captures her elegance and strength, with her attire reflecting a blend of exotic beauty and grace. "
                f"The woman is in the <img><|image_1|></img>."
            )
            prompts.append(prompt)
            prompt = (
                f"The woman wore large golden earrings and a thick necklace. Her body was glittering with golden sequins. She leaned against a black panther. The panther had sharp eyes and its yellow eyes were very striking. The background was water surface. "
                f"The woman is in the <img><|image_1|></img>."
            )
            prompts.append(prompt)

    # 感恩节女
    # 女1：This photo depicts a festive scene filled with a warm atmosphere: A lady is wearing a red long-sleeved top, adorned with a delicate necklace, and smiling as she holds a beautifully plated golden-brown roasted turkey. 
    #       Around the turkey are yellow mini pumpkins and green decorative dishes, along with some lemon slices. The background is a warm yellow light, and the other attendees of the party can be vaguely seen.
    #       The overall scene creates a profound sense of warmth for the Thanksgiving family reunion, conveying a joyful and warm festive atmosphere.The woman is in the <img><|image_1|></img>.
    # 女2：This photo is filled with a strong sense of the autumn harvest: A woman is wearing a white blouse, squatting in a pile of pumpkins, holding a round and orange large pumpkin with both hands, and her face is brimming with a bright smile.
    #       Around her are various colors and sizes of pumpkins, including orange, light yellow and white. There are also wooden boxes with flowers and a red cart filled with fruits beside her. The background seems to be a pumpkin farm or market.
    #       The overall picture is dominated by warm tones, conveying a leisurely and pleasant atmosphere of autumn picking, fully expressing the joy of harvest and the delightful enjoyment of the season. The woman is in the <img><|image_1|></img>.
    # 女3：In the photo, the woman is wearing an orange coat and a white scarf. She is pushing a handcart filled with various pumpkins, standing beside the autumn fence path. The sunlight filters through the trees, casting dappled shadows on the ground.
    #       Fallen leaves dot the ground, exuding a warm and festive atmosphere of harvest. The woman is in the <img><|image_1|></img>.
    # 女4：Half-body photo. This photo creates a cozy autumn atmosphere: A lady is wearing a colorful chunky knitted hooded sweater, with warm yellow, red-brown, blue-gray and other tones interwoven, giving it a thick texture and a retro feel. 
    #       She is holding a glass of white wine. The background is the golden autumn forest, with fallen leaves covering the ground.
    #       The warm-toned light and shadow make the entire scene full of a lazy and leisurely autumn mood, as if enjoying an outdoor autumn bittersweet moment, conveying a relaxed and carefree sense of life. The woman is in the <img><|image_1|></img>.
    elif template == "thanksgiving-lady":
        prompt = (
            f"This photo depicts a festive scene filled with a warm atmosphere: A {call} is wearing a red long-sleeved top, adorned with a delicate necklace, and smiling as {pronoun} holds a beautifully plated golden-brown roasted turkey. "
            f"Around the turkey are yellow mini pumpkins and green decorative dishes, along with some lemon slices. The background is a warm yellow light, and the other attendees of the party can be vaguely seen. "
            f"The overall scene creates a profound sense of warmth for the Thanksgiving family reunion, conveying a joyful and warm festive atmosphere. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"This photo is filled with a strong sense of the autumn harvest: A {call} is wearing a white blouse, squatting in a pile of pumpkins, holding a round and orange large pumpkin with both hands, and {possessive} face is brimming with a bright smile."
            f"Around {with_} are various colors and sizes of pumpkins, including orange, light yellow and white. There are also wooden boxes with flowers and a red cart filled with fruits beside {with_}. The background seems to be a pumpkin farm or market. "
            f"The overall picture is dominated by warm tones, conveying a leisurely and pleasant atmosphere of autumn picking, fully expressing the joy of harvest and the delightful enjoyment of the season. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"In the photo, the {call} is wearing an orange coat and a white scarf. {pronoun} is pushing a handcart filled with various pumpkins, standing beside the autumn fence path. The sunlight filters through the trees, casting dappled shadows on the ground. "
            f"Fallen leaves dot the ground, exuding a warm and festive atmosphere of harvest. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"Half-body photo. This photo creates a cozy autumn atmosphere: A {call} is wearing a colorful chunky knitted hooded sweater, with warm yellow, red-brown, blue-gray and other tones interwoven, giving it a thick texture and a retro feel. "
            f"{pronoun} is holding a glass of white wine. The background is the golden autumn forest, with fallen leaves covering the ground. "
            f"The warm-toned light and shadow make the entire scene full of a lazy and leisurely autumn mood, as if enjoying an outdoor autumn bittersweet moment, conveying a relaxed and carefree sense of life. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)  

    # 感恩节婴儿
    # 婴儿1：This photo is filled with the warm and playful atmosphere of the autumn Thanksgiving. A baby is wearing a fluffy costume in the shape of a turkey, with a turkey hat adorned with colorful feathers on his head, sitting in a brown pot.
    #       Surrounding him are orange pumpkins, colorful autumn leaves and bright flowers. On the brick wall in the background, there are various colors of pumpkins piled up.
    #       The overall picture exudes a rich festive atmosphere, being both adorable and soothing.The baby is in the <img><|image_1|></img>.
    # 婴儿2：The baby is wearing a white chef's hat and is dressed in a Thanksgiving-themed apron, which is decorated with elements such as autumn leaves and pumpkins. Holding a plate of roasted turkey, the baby fully demonstrates the festive participation.
    #       The background is a kitchen, and the table is set with Thanksgiving delicacies like bell peppers, pumpkins, and mixed dishes, creating a warm scene of the family preparing the holiday feast together.The baby is in the <img><|image_1|></img>.
    # 婴儿3：The baby is wearing an orange and brown checkered sweater, paired with brown trousers, an orange knitted cap (with pompoms), a scarf of the same color, and brown leather boots.
    #       The overall style of the outfit is warm and full of autumn charm. The background is a cozy indoor family scene, with lit candles on the wooden dining table, orange and white pumpkins, and autumn leaves decorations.
    #       In the distance, one can see family members sitting together, creating a very warm and united atmosphere.The baby is in the <img><|image_1|></img>.
    # 婴儿4：The baby is dressed in a light-colored one-piece garment and is lying in a woven basket. Surrounding it are various colorful mini pumpkins in shades of orange, white, and green.
    #       The bottom of the basket is covered with white fabric, and the background resembles a white plush material. The scene is warm and gentle.The baby is in the <img><|image_1|></img>.
    elif template == "thanksgiving-baby":
        if info["age"] <= 5:
            call = "baby"
        prompt = (
            f"This photo is filled with the warm and playful atmosphere of the autumn Thanksgiving. A {call} is wearing a fluffy costume in the shape of a turkey, with a turkey hat adorned with colorful feathers on {possessive} head, sitting in a brown pot. "
            f"Surrounding {with_} are orange pumpkins, colorful autumn leaves and bright flowers. On the brick wall in the background, there are various colors of pumpkins piled up. "
            f"The overall picture exudes a rich festive atmosphere, being both adorable and soothing. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"The {call} is wearing a white chef's hat and is dressed in a Thanksgiving-themed apron, which is decorated with elements such as autumn leaves and pumpkins. Holding a plate of roasted turkey, the {call} fully demonstrates the festive participation. "
            f"The background is a kitchen, and the table is set with Thanksgiving delicacies like bell peppers, pumpkins, and mixed dishes, creating a warm scene of the family preparing the holiday feast together. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"The {call} is wearing an orange and brown checkered sweater, paired with brown trousers, an orange knitted cap (with pompoms), a scarf of the same color, and brown leather boots. "
            f"The overall style of the outfit is warm and full of autumn charm. The background is a cozy indoor family scene, with lit candles on the wooden dining table, orange and white pumpkins, and autumn leaves decorations."
            f"In the distance, one can see family members sitting together, creating a very warm and united atmosphere.The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
        prompt = (
            f"The {call} is dressed in a light-colored one-piece garment and is lying in a woven basket. Surrounding it are various colorful mini pumpkins in shades of orange, white, and green. "
            f"The bottom of the basket is covered with white fabric, and the background resembles a white plush material. The scene is warm and gentle. The {call} is in the <img><|image_1|></img>."
        )
        prompts.append(prompt)
    return prompts

def inference_onmigen(prompt, input_images, height, width, template):
    if prompt == "":
        return None

    if template in ["microphone", "cartoon", "identification", "workplace", "exotic"]:
        correct_faces = 1
    elif template == "photo":
        correct_faces = 3
    else:
        correct_faces = 2

    img_guidance_scale = 1.6
    images = pipe(
        prompt=prompt,
        input_images=input_images,
        height=height//16*16, 
        width=width//16*16,
        guidance_scale=3, #2.5 
        img_guidance_scale=img_guidance_scale,
        num_inference_steps=35,
        seed=0
    )
    image = images[0]
    faces = app_gen.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    input_infos = [analysis_face(input_image) for input_image in input_images]
    faces_similarity = [get_face_similarity(face, input_infos) for face in faces]
    print(f"faces_gender: {[face.sex for face in faces]}")
    print(f"faces_similarity: {faces_similarity}")
    retry = 0
    while(len(faces) > correct_faces or len(faces_similarity) == 0 or min(faces_similarity) < 0.25):
        if len(faces) == correct_faces and len(faces_similarity) > 0 and min(faces_similarity) < 0.25:
            img_guidance_scale += 0.2
        images = pipe(
            prompt=prompt,
            input_images=input_images,
            height=height//16*16,
            width=width//16*16,
            guidance_scale=3, #2.5
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=35,
        )
        image = images[0]
        faces = app_gen.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        faces_similarity = [get_face_similarity(face, input_infos) for face in faces]
        print(f"faces_gender: {[face.sex for face in faces]}")
        print(f"faces_similarity: {faces_similarity}")
        retry += 1
        if retry >= 5:
            break
    image = image.resize((width, height))
    return image




if __name__ == "__main__":
    template = "microphone"
    input_images = ["./imgs/lw/facefun_muban/baozinvlang3.jpg"]
    prompt = generate_prompt(input_images[0], "", template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width, template)
    image.save("./imgs/sidatian/microphone-0403.png")

    template = "horse"
    input_images = ["./imgs/lw/facefun_muban/baozinvlang3.jpg", "./samples/horse.jpg"]
    prompt = generate_prompt(input_images[0], input_images[1], template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width, template)
    image.save("./imgs/sidatian/horse-0220.png")

    template = "kangaroo"
    input_images = ["./imgs/lw/facefun_muban/baozinvlang3.jpg", "./samples/kangaroo.jpg"]
    prompt = generate_prompt(input_images[0], input_images[1], template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width, template)
    image.save("./imgs/sidatian/kangaroo-0220.png")

    template = "elephant"
    input_images = ["./imgs/lw/facefun_muban/baozinvlang3.jpg"]
    prompt = generate_prompt(input_images[0], "", template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width, template)
    image.save("./imgs/sidatian/elephant-0315.png")

    template = "snowflakes"
    input_images = ["./imgs/lw/facefun-0417/4.jpg", "./imgs/lw/facefun-0417/3.jpg"]
    prompt = generate_prompt(input_images[0], input_images[1], template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width, template)
    image.save("./imgs/sidatian/snowflakes-0506.png")

    template = "photo"
    input_images = ["./imgs/lw/facefun-0417/4.jpg", "./imgs/lw/facefun-0417/3.jpg", "./imgs/lw/facefun-0417/33.jpg"]
    prompt = generate_prompt_for_3(input_images[0], input_images[1], input_images[2], template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width, template)
    image.save("./imgs/sidatian/photo-0315.png")

    template = "exotic"
    image_file = "./imgs/lw/styleme-0310/8.png"
    height = 960
    width = 720
    prompts = generate_prompts_styleme(image_file, template)
    for index, prompt in enumerate(prompts):
        print(prompt)
        image = inference_onmigen(prompt, [image_file], height, width, template)
        image.save(f"./imgs/sidatian/styleme-0312-{index}.png")
    