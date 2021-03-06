import os
import time
import math
import cv2
import numpy as np
import argparse
import requests
import sys
import tqdm
import craft_utils
import imgproc

from typing import Tuple, Union
from numpy import median

from ppocr.utils import utility
from ppocr.utils import logging
from PIL import Image
from collections import OrderedDict
from pdf2image import convert_from_path
from Get_Angle import getAngleBetweenPoints, getPoint_Center, getPointRotate
from Get_IntersectionPoint import getIntersection
from Get_Distance import getDistance
from craft import CRAFT

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from craft import CRAFT

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

logger = logging.get_logger()

config = Cfg.load_config_from_file('config.yml')
detector = Predictor(config)

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + \
        abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + \
        abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    try:
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(255, 255, 255))
    except:
        print('none')


def draw(img, line):
    img = cv2.line(img, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), color=[255, 0, 0],
                   thickness=2)
    img = cv2.line(img, (int(line[1][0]), int(line[1][1])), (int(line[2][0]), int(line[2][1])), color=[255, 0, 0],
                   thickness=2)
    img = cv2.line(img, (int(line[2][0]), int(line[2][1])), (int(line[3][0]), int(line[3][1])), color=[255, 0, 0],
                   thickness=2)
    img = cv2.line(img, (int(line[3][0]), int(line[3][1])), (int(line[0][0]), int(line[0][1])), color=[255, 0, 0],
                   thickness=2)
    return img


def Crop_Image(img, line, angle):
    remain_x = 5
    remain_y = 3
    h, w = img.shape[0], img.shape[1]

    if int(line[0][0]) - remain_x < 0:
        x0 = 0
    else:
        x0 = int(line[0][0]) - remain_x
    if int(line[0][1]) - remain_y < 0:
        y0 = 0
    else:
        y0 = int(line[0][1]) - remain_y
    if int(line[1][0]) + remain_x > w:
        x1 = w
    else:
        x1 = int(line[1][0]) + remain_x
    if int(line[1][1]) - remain_y < 0:
        y1 = 0
    else:
        y1 = int(line[1][1]) - remain_y
    if int(line[2][0]) + remain_x > w:
        x2 = w
    else:
        x2 = int(line[2][0]) + remain_x
    if int(line[2][1]) + remain_y > h:
        y2 = h
    else:
        y2 = int(line[2][1]) + remain_y
    if int(line[3][0]) - remain_x < 0:
        x3 = 0
    else:
        x3 = int(line[3][0]) - remain_x
    if int(line[3][1]) + remain_y > h:
        y3 = h
    else:
        y3 = int(line[3][1]) + remain_y

    pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    crop = img[y:y + h, x:x + w]
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    pts = pts - pts.min(axis=0)
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(crop, crop, mask=mask)

    # (4) add the white background
    bg = np.ones_like(crop, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst
    rotated = rotate(dst2, angle, (0, 0, 0))
    bg = rotate(bg, angle, (0, 0, 0))

    _, binary = cv2.threshold(bg, 254, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    x, y, width, height = cv2.boundingRect(cnts[0])

    result = rotated[y:y + height, x:x + width]
    return result, height, width


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights\craft_mlt_25k.pth',
                    type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7,
                    type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float,
                    help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4,
                    type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280,
                    type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float,
                    help='image magnification ratio')
parser.add_argument('--poly', default=False,
                    action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False,
                    action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='test',
#                     type=str, help='folder path to input images')
parser.add_argument('--refine', default=False,
                    action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights\craft_refiner_CTW1500.pth',
                    type=str, help='pretrained refiner model')

args = parser.parse_args()

# """ For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def npf32u8(np_arr):
    # intensity conversion
    if str(np_arr.dtype) != 'uint8':
        np_arr = np_arr.astype(np.float32)
        np_arr -= np.min(np_arr)
        np_arr /= np.max(np_arr)  # normalize the data to 0 - 1
        np_arr = 255 * np_arr  # Now scale by 255
        np_arr = np_arr.astype(np.uint8)
    return np_arr


def opencv2pil(opencv_image):
    # convert numpy array type float32 to uint8
    opencv_image_rgb = npf32u8(opencv_image)
    # convert numpy array to Pillow Image Object
    pil_image = Image.fromarray(opencv_image_rgb)
    return pil_image


def init(folder1, folder2, folder3, folder4):
    try:
        for file in os.listdir(folder1):
            os.remove(folder1 + file)
    except:
        pass
    try:
        for file in os.listdir(folder2):
            os.remove(folder2 + file)
    except:
        pass
    try:
        for file in os.listdir(folder3):
            os.remove(folder3 + file)
    except:
        pass
    try:
        for file in os.listdir(folder4):
            os.remove(folder4 + file)
    except:
        pass


def sorter(item):
    return item[0][0]


# def crop2multiImage(image):
#     # get width, height of image
#     H = image.shape[0]
#     W = image.shape[1]

#     detal_w = W//10
#     detal_h = H//10

#     # crop img
#     for i in range(10):
#         for j in range(10):
#             x1 = j*detal_w
#             y1 = i*detal_h
#             x2 = (j+1)*detal_w
#             y2 = (i+1)*detal_h
#             print("--> (x1,y1) = ", (x1, y1))
#             print("--> (x2,y2) = ", (x2, y2))

#             crop_img = image[y1:y2, x1:x2]

#     return crop_img


# def optimize_result(result,x1,y1):
#     new_result = []

#     for i in range(len(result)):
#         text_box = result[i]

#         lefttop = text_box[0]
#         righttop = text_box[1]
#         botright = text_box[2]
#         botleft = text_box[3]

#         new_lefttop = [lefttop[0] + x1,lefttop[1] + y1]
#         new_righttop = [righttop[0] + x1,righttop[1] + y1]
#         new_botright = [botright[0] + x1,botright[1] + y1]
#         new_botleft = [botleft[0] + x1,botleft[1] + y1]

#         new_text_box = [new_lefttop,new_righttop,new_botright,new_botleft]

#         new_result.append(new_text_box)

#     return new_result


def run(image_dir):

    init('./line_cut/', './pdf_image/', './results/images/', './results/texts/')

    # load model text detection
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(
            args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(
                copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(
                torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    # load data
    if '.pdf' in image_dir:
        print('________pdf________')
        images = convert_from_path(
            image_dir, poppler_path='poppler-0.68.0/bin')
        for i in range(len(images)):
            images[i].save('./pdf_image/page_' + str(i + 1) + '.jpg', 'JPEG')
        image_file_list = utility.get_image_file_list('.\\pdf_image')

    else:
        if image_dir.startswith('http'):
            download_with_progressbar(image_dir, 'tmp.jpg')
            image_file_list = ['tmp.jpg']
        else:
            image_file_list = utility.get_image_file_list(image_dir)
        if len(image_file_list) == 0:
            logger.error('no images find in {}'.format(image_dir))

    start = time.time()
    count_1 = 1
    count_2 = 1
    result_ocr = ''

    for image_path in image_file_list:
        f = open('./results/text/text_' + str(count_2) +
                 '.txt', 'w', encoding='utf-8')
        # img = imgproc.loadImage(image_path)
        img = cv2.imread(image_path)
        w_0, h_0 = img.shape[:2]

        # deskew
        bboxes, polys, score_text = test_net(
            net, img, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        result_0 = bboxes
        list_angle = []
        if result_0 is not None:
            for line in result_0:
                angle = getAngleBetweenPoints(line)
                list_angle.append(int(angle))
            list_angle.sort()
            angle_skew = median(list_angle)
            img = rotate(img, angle_skew, (0, 0, 0))
            try:
                cv2.imwrite('./results/deskew.jpg', img)
            except:
                continue
        # crop ???nh theo angle_skew
        w, h = img.shape[:2]
        result_1 = []
        center_0 = [int(w_0/2), int(h_0/2)]
        center = [int(w/2), int(h/2)]
        angle_skew = math.pi*angle_skew/180

        if result_0 is not None:
            for line in result_0:
                line_1 = []
                for point in line:
                    point = getPoint_Center(point, center_0)
                    point = getPointRotate(point, angle_skew, center)
                    line_1.append(point)
                result_1.append(line_1)

        img1 = img.copy()

        # result_1 = []

        # # chia ???nh
        # w = img.shape[1]
        # h = img.shape[0]

        # split_w = w//10
        # split_h = h//10

        # for i in range(10):
        #     for j in range(10):
        #         x1 = j * split_w
        #         y1 = i * split_h
        #         x2 = (j+1)*split_w
        #         y2 = (i+1)*split_h

        #         crop_img = img[y1:y2, x1:x2]

        #         bboxes_1, polys_1, score_text_1 = test_net(net, crop_img, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        #         item_result = optimize_result(bboxes_1,x1,y1)

        #         for k in range(len(item_result)):
        #             text_box = item_result[k]
        #             result_1.append(text_box)

        # bboxes_1, polys_1, score_text_1 = test_net(
        #     net, img, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # result_1 = bboxes_1
        list_point = []
        list_flag = []
        for line in result_1:
            i_point = getIntersection([line[0], line[2]], [line[1], line[3]])
            list_point.append(i_point)
            list_flag.append(0)
        temp = []
        # temp = np.empty((0, 2), dtype=np.float, order="C")
        try:
            temp.append(result_1[0])
        except:
            continue
        list_flag[0] = 1
        i = 0
        end_ = False
        while i < len(result_1):
            if list_flag[i] == 1:
                if i + 1 < len(result_1):
                    h = (getDistance(result_1[i][0], result_1[i][3]) +
                         getDistance(result_1[i][1], result_1[i][2])) / 4
                    if list_point[i + 1][1] <= list_point[i][1] + 1.2 * h and list_point[i + 1][1] >= list_point[i][
                            1] - 1.2 * h:
                        list_flag[i + 1] = 1
                        temp.append(result_1[i + 1])
                        # temp = np.append(temp, [np.array(result_1[i + 1])], axis=0)
            else:

                temp = sorted(temp, key=sorter)

                for line in temp:
                    img1 = draw(img1, line)
                    angle = getAngleBetweenPoints(line)
                    crop, crop_h, crop_w = Crop_Image(img, line, angle)
                    cv2.imwrite('./line_cut/' + str(count_1) + '.jpg', crop)
                    gray = opencv2pil(crop)

                    try:
                        # start1 = time.time()
                        text, prob = detector.predict(gray,
                                                      return_prob=True)  # mu???n tr??? v??? x??c su???t c???a c??u d??? ??o??n th?? ?????i return_prob=True
                        # end1 = time.time()
                        if str(prob) != 'nan':
                            if float(prob) >= 0.6:
                                f.write(text + '\t')
                                result_ocr += text + '\t'
                                print(text)
                    except:
                        continue
                    count_1 += 1
                f.write('\n')
                result_ocr += '\n'
                temp.clear()
                if end_:
                    break
                if i != len(result_1) - 1:
                    list_flag[i] = 1
                else:
                    end_ = True
                temp.append(result_1[i])
                i -= 1
            i += 1
        cv2.imwrite('./results/images/text_detection_' +
                    str(count_2) + '.jpg', img1)
        count_2 += 1
        f.write('\n')
        result_ocr += '\n'
        f.close()
        print("Done!!!!!!!!!!")
    folder_line_cut = './results/line_cut/'
    for image in os.listdir(folder_line_cut):
        os.remove(folder_line_cut + image)
    os.remove('./results/deskew.jpg')
    end = time.time()
    print('Time: %5.3f' % (end - start), 's')
    return result_ocr.strip()


if __name__ == '__main__':
    # image_dir = 'C:/Users/tuyen/Desktop/Data/CV/Meyland/11-1-21'
    image_dir = 'test'
    run(image_dir)
