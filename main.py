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
    template_list = [
        "standing", "wedding", "graduation", "pet", "rock", "photograph", "broom", "cheetah", "muscles", "beach", "trophy", "egyptian", "balloons",
        "titanic", "redcar", "brazil", "wallstreet", "convertible", "doctor", "rain", "horse", "kangaroo", "eiffeltower", "wall", "pregnant", "seabed", "mother", "gift", "microphone"]
    assert template in template_list,\
        f"template {template} not in template_list(standing, wedding, graduation, pet, rock, photograph, broom, cheetah, muscles, beach, trophy, egyptian, balloons," \
        "titanic, redcar, brazil, wallstreet, convertible, doctor, rain, horse, kangaroo, eiffeltower, wall, pregnant, seabed, mother, gift, microphone)"

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

    # 埃及：A man and a woman stand in front of an ancient Egyptian temple, bathed in the golden sunlight.
    #       The man is tall and strong, with long golden brown hair tied into a low ponytail, wearing a white knee-length skirt, a wide gold belt around his waist, and gold ornaments on his arms, with a majestic temperament.
    #       The woman has black hair that reaches her shoulders, and her eyeliner outlines her mysterious charm. She wears a close-fitting white dress and a gold collar with sapphires around her neck, just like an ancient Egyptian queen. 
    #       They stand side by side, with deep eyes, and the background is the majestic temple and pyramids. The setting sun casts afterglow, making the whole picture full of mystery and solemnity.
    elif template == "egyptian":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} and a {} stand in front of an ancient Egyptian temple, bathed in the golden sunlight. ".format(call_1, call_2)
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
            prompt = "The {} and the {} stood together, smiling towards the camera. The background was a serene outdoor scene, with a vintage red sports car behind them, neatly trimmed green plants and a white fence. ".format(call_1, call_2)
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
            prompt = "The {} and the {} stood together, behind them was a huge floor-to-ceiling window, and outside the window were skyscrapers with neon lights twinkling. ".format(call_1, call_2)
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
            prompt = "A {} and a {}, both wearing white coats and with a stethoscope around their necks, stood inside a hospital. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 雨中：Men and women stood on a street, the rain pouring down on them, their clothes soaked.
    elif template == "rain":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "The {} and the {} stood on a street, the rain pouring down on them, their clothes soaked. ".format(call_1, call_2)
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
            prompt = "A {} and a {} stood together under the Eiffel Tower and the cherry blossom trees. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 壁咚：The man and the woman are standing together. There is a wall beside the woman.
    elif template == "wall":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "The {} and the {} are standing together. ".format(call_1, call_2)
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
            prompt = "The woman is pregnant and the man is standing beside her. "
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 海底：This is a photo taken from the seabed. A couple of people are in the blue underwater world. The man is gently embracing the woman.
    #       Around them, some beautiful dolphins are swimming gracefully, and various kinds of small fish are darting and swimming among the coral reefs.
    #       The bubbles rising from the bottom of the water add a sense of movement to the picture.
    elif template == "seabed":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "This is a photo taken from the seabed. A couple of people are in the blue underwater world. "
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
            prompt = "A {} and a {} stand together, with the background being the blue sky and white clouds outside. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 礼物：A man and a woman stand together.
    elif template == "gift":
        if len(call_1) == 0 or len(call_2) == 0:
            prompt = ""
        else:
            prompt = "A {} and a {} stand together. ".format(call_1, call_2)
            prompt = prompt + "The {} is in the <img><|image_1|></img> and the {} is in the <img><|image_2|></img>.".format(call_1, call_2)

    # 麦克风：A woman was standing on a podium, with a microphone in front of her.
    elif template == "microphone":
        if len(call_1) == 0:
            prompt = ""
        else:
            prompt = "A {} was standing on a podium, with a microphone in front of her. ".format(call_1)
            prompt = prompt + "The {} is in the <img><|image_1|></img>.".format(call_1)
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
    template = "horse"
    input_images = ["./imgs/lw/facefun_muban/baozinvlang3.jpg", "./samples/horse.jpg"]
    prompt = generate_prompt(input_images[0], input_images[1], template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width)
    image.save("./imgs/sidatian/horse-0220.png")

    template = "kangaroo"
    input_images = ["./imgs/lw/facefun_muban/baozinvlang3.jpg", "./samples/kangaroo.jpg"]
    prompt = generate_prompt(input_images[0], input_images[1], template)
    print(prompt)
    height = 960
    width = 720
    image = inference_onmigen(prompt, input_images, height, width)
    image.save("./imgs/sidatian/kangaroo-0220.png")
