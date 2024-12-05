import cv2
import numpy as np
import os
from PIL import Image

def create_random_mask(image_path, output_mask_path, mask_size=(50, 50), num_masks=5):
    """
    특정 이미지를 입력으로 받아 랜덤한 마스크를 생성하고 저장합니다.

    :param image_path: 입력 이미지 경로
    :param output_mask_path: 생성된 마스크 이미지 저장 경로
    :param mask_size: 각 마스크 영역의 크기 (width, height)
    :param num_masks: 생성할 랜덤 마스크의 개수
    """
    print(image_path)
    # 원본 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to load the image. Please check the path.")

    # 이미지와 동일한 크기의 빈 마스크 생성 (검은색)
    mask = np.zeros_like(image, dtype=np.uint8)

    # 이미지 크기 가져오기
    h, w, _ = image.shape

    # 랜덤한 위치에 흰색 박스 마스크 생성
    for _ in range(num_masks):
        top_left_x = np.random.randint(0, w - mask_size[0])
        top_left_y = np.random.randint(0, h - mask_size[1])
        bottom_right_x = top_left_x + mask_size[0]
        bottom_right_y = top_left_y + mask_size[1]

        # 흰색으로 마스크 영역 지정
        mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = (255, 255, 255)

    # 마스크 저장
    cv2.imwrite(output_mask_path, mask)

    # PIL로 보기 쉽게 변환하여 반환
    return Image.fromarray(mask)
if __name__ == '__main__':
    # 사용 예시
    image_path = os.path.join(os.path.dirname(__file__), 'images/test.png') # 원본 이미지 경로
    output_mask_path = os.path.join(os.path.dirname(__file__), 'output/output1.png')  # 생성된 마스크 저장 경로

    # 랜덤 마스크 생성 (50x50 크기의 마스크를 5개 생성)
    random_mask = create_random_mask(image_path, output_mask_path, mask_size=(200, 200), num_masks=1)