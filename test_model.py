import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model
import os
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'



def get_args(): #콘솔로부터 인자를 받아오는 함수
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):   #이미지를 받아오는 함수
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main(): 
    # get_args() 함수로부터 인자를 받아옴
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    val_noise_model = get_noise_model(args.test_noise_model)
    model = get_model(args.model)
    model.load_weights(weight_file)

    if args.output_dir: #out_path를 입력받았다면
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:  #image_path 내에 있는 모든 이미지 파일을 순차적으로 작업함
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape
        image = image[:(h // 16) * 16, :(w // 16) * 16] 
        h, w, _ = image.shape
        
        out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
        noise_image = val_noise_model(image)
        pred = model.predict(np.expand_dims(noise_image, 0))
        denoised_image = get_image(pred[0])
 
        #out_image에 imgae, noise_image, denoised_image를 넣음.(사용하지 않음)
        execution_path = os.getcwd()
        out_image[:, :w] = image
        out_image[:, w:w * 2] = noise_image
        out_image[:, w * 2:] = denoised_image
        #cv2.imwrite("denoised.jpg", out_image)
        cv2.imwrite("denoised.jpg", denoised_image) #denoised된 이미지를 기본경로에 저장
        #img = cv2.imread(execution_path, "denoised.jpg")
        #경로 내에 존재하는 denoised.jpg파일을 tesseract로 불러옴
        data = pytesseract.image_to_string(Image.open('denoised.jpg'), lang='eng')
        #불러온 image 내의 문자열을 data 변수에 string으로 입력
        f = open("result.txt", 'w', encoding='UTF8')
        f.write(data)   #result.txt파일 안에 해당 string을 저장하고 기본 경로에 저장.
        f.close()

        
        if args.output_dir: #output_path를 입력받았다면 그 경로에 저장
            cv2.imwrite("denoised.png", out_image)
        #else:
            #cv2.imshow("result", out_image)

if __name__ == '__main__':
    main()
