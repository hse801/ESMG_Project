# Test ocr with google vision api
# TOEFL book images taken manually

import io
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./ocr-project.json"
# Imports the Google Cloud client library
from google.cloud import vision
import glob
import numpy as np
from PIL import Image
import cv2
import googletrans
from keybert import KeyBERT

"""

영어 학습 보조 자료 제작 서비스

1. yolo v5를 통해 지문 영역 detection
2. yolo v5의 prediction result에서 지문 영역의 좌표 값을 추출
3. 포맷을 변환하여 해당 영역의 이미지만 크롭
4. 크롭된 이미지를 google vision api를 통해 ocr
5. ocr을 통해 얻은 텍스트 정보를 googletrans api를 이용하여 번역본 생성
6. keyBERT를 이용하여 키워드 추출
7. 키워드를 빈칸으로 대체하여 빈칸 문제 생성

"""


def convert_coord(label_path):
    # convert (class, CenterX, CenterY, W, H) to (StartX, StartY, endX, endY)
    f = open(label_path, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        values = line.split(' ')
        print(f'values = {values}')
        if values[0] == '0':
            values = list(np.float_(values))
            # values = map(float, values)
            w = values[3]
            h = values[4]
            start_x = values[1] - 0.5 * w
            start_y = values[2] + 0.5 * h
            end_x = values[1] + 0.5 * w
            end_y = values[2] - 0.5 * h

            print(f'converted({values[0]}, {values[1]}, {w}, {h}) to ({start_x:.4}, {start_y:.4}, {w}, {h})')

            return start_x, start_y, end_x, end_y
        else:
            return
    f.close()


def translate(text):
    # 추출한 text를 번역
    # googletrans api 이용

    translator = googletrans.Translator()
    result = translator.translate(text, src='en', dest='ko')
    print(f'--------------------------------Translated Text--------------------------------')
    print(result.text)
    f = open('E:/HSE/2021-2/Project/results/translated.txt', 'w')
    f.write('<통암기 유형>\n')
    f.write(result.text)
    f.close()


def keyword(text):
    # Keyword extraction using KeyBERT model
    # 빈칸 뚫기 유형 문제 생성 위함

    kw_model = KeyBERT()
    # keywords = kw_model.extract_keywords(text, use_maxsum=True, nr_candidates=20, top_n=5)
    keywords = kw_model.extract_keywords(text)
    # print(f'keywords type = {type(keywords)}, len = {len(keyword)}')
    print(keywords)
    for key in keywords:
        text = text.replace(key[0], '____________')
    print('Text with Blank:')
    print(text)
    f = open('E:/HSE/2021-2/Project/results/blank.txt', 'w')
    f.write('<빈칸 유형>\n')
    for idx, key in enumerate(keywords):
        text = text.replace(key[0], f'_____{idx + 1}_____')
    f.write(text)
    f.write('\n\n[Answer]\n\n')
    print(f'[Answer]')
    for idx, key in enumerate(keywords):
        print(f'Question {idx + 1}: {key[0]}')
        f.write(f'Question {idx + 1}: {key[0]}\n')
    f.close()


def ocr(img_path, label_path):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    img = Image.open(img_path)
    print(f'img type = {type(img)}, shape = {np.shape(img)}')
    img_h, img_w = np.shape(img)[0], np.shape(img)[1]
    start_x, start_y, end_x, end_y = convert_coord(label_path)
    start_x *= img_w
    end_x *= img_w
    start_y *= img_h
    end_y *= img_h
    print(f'crop to ({start_x:.4}, {start_y:.4}, {end_x}, {end_y})')
    area = (start_x, end_y, end_x, start_y)
    cropped_img = img.crop(area)

    img_byte_arr = io.BytesIO()
    cropped_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # img.show()
    # cropped_img.show()
    #
    # # The name of the image file to annotate
    # file_name = os.path.abspath(r'./Toefl/Toefl-01.jpg')
    #
    # file_name = img_path
    # # Loads the image into memory
    # with io.open(file_name, 'rb') as image_file:
    #     content = image_file.read()

    # content = cropped_img

    image = vision.Image(content=img_byte_arr)
    # print(f'cropped type = {type(cropped_img)}, content = {type(content)}, image = {type(image)}')

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    # print(f'image = {image}')
    # print(f'response = {response}')
    # print('Labels:')
    # for label in labels:
    #     print(label.description)

    # Performs text detection on the image file
    response = client.text_detection(image=image)
    texts = response.text_annotations

    my_text = texts[0].description
    my_text = my_text.replace('\n', ' ')

    print('Texts:')
    # for text in texts:
    print(my_text)

    f = open('E:/HSE/2021-2/Project/results/ocr_text.txt', 'w')
    f.write('<Text>\n')
    f.write(my_text)
    f.close()
    return my_text


img_path = glob.glob('E:/HSE/2021-2/Project/data/train/Toefl-06.jpg')


for img in img_path:
    img_name = os.path.split(img)
    label_name = img_name[-1].replace('jpg', 'txt')
    label_path = f'E:/HSE/2021-2/Project/yolov5/runs/detect/exp5/labels/{label_name}'
    # ocr 함수에서 지문 text 추출
    text = ocr(img_path=img, label_path=label_path)
    # 추출된 지문의 번역본 제공
    translate(text=text)
    # 키워드 추출하여 빈칸 채우기 유형 문제 제공
    keyword(text=text)
    break


