import cv2
import numpy as np
import os
import copy
import math


def findNearContour(edges, img_org, x, y, hight=20):
    try:
        if y - hight > 0 and y + hight < 256 and x - hight > 0 and x + hight < 256:
            img_acx = edges[y - hight: y + hight, x - hight: x + hight]
            img_crop = img_org[y - hight: y + hight, x - hight: x + hight]
            img_acx = copy.copy(img_acx)
            corners1 = cv2.goodFeaturesToTrack(
                img_acx, 10, minDistance=20, qualityLevel=0.1)
            corners1 = np.int0(corners1)
            img_acx = cv2.cvtColor(img_acx, cv2.COLOR_GRAY2BGR)
            for index_c in corners1:
                i_x, i_y = index_c.ravel()
            if len(corners1) == 1:
                cv2.circle(img_acx, (i_x, i_x), 1, (255, 0,  0), 2)
            if len(corners1) == 1:
                return True
            return False
    except:
        return False


def distanse(x0, y0, x1, y1):
    return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)


def remove_short_line(x0, y0, x1, y1, w, h):
    small_dis = h
    if w > h:
        small_dis = w
    dis = distanse(x0, y0, x1, y1)
    # remove
    if dis < (small_dis / 10):
        return True
    return False


def swap(swap1, swap2):
    return swap2, swap1


def conner_findder(img_org, x, y, leng=10):
    """
    :param img_org: original image
    :param x: x conner are
    :param y: y conner are
    :param leng: w, h will be set
    :return: ROI and ROI are conner or
    """
    img_org = copy.copy(img_org)
    h, w, c = img_org.shape

    y_up = y - leng
    if y - leng < 0:
        y_up = 0
    y_down = y + leng
    if y + leng > h:
        y_down = h
    x_up = x - leng
    if x - leng < 0:
        x_up = 0
    x_down = x + leng
    if x + leng > w:
        x_down = w

    if y_up > y_down:
        y_up, y_down = swap(y_up, y_down)

    if x_up > x_down:
        x_up, x_down = swap(x_up, x_down)

    ROI = img_org[y_up: y_down, x_up: x_down]
    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    dlines = lsd.detect(ROI_gray)
    h_ROI, w_ROI = y_down - y_up, x_down - x_up
    vec_list = np.empty((0, 2), order='C')
    list_all_point = np.empty((0, 2), dtype=np.uint8, order='C')

    for dline in dlines[0]:
        x0_c = int(round(dline[0][0]))
        y0_c = int(round(dline[0][1]))
        x1_c = int(round(dline[0][2]))
        y1_c = int(round(dline[0][3]))
        if distanse(x0_c, y0_c, x1_c, y1_c) > (h_ROI / 3):

            vec = vector_normalization(x0_c, y0_c, x1_c, y1_c)

            if len(vec_list) == 0:
                vec_list = np.append(vec_list, [vec], axis=0)
                cv2.circle(ROI, (x0_c, y0_c), 1, (0, 0, 255), 2)
                cv2.circle(ROI, (x1_c, y1_c), 1, (0, 0, 255), 2)
                list_all_point = np.append(
                    list_all_point, [[x0_c, y0_c]], axis=0)
                list_all_point = np.append(
                    list_all_point, [[x1_c, y1_c]], axis=0)
            else:
                count_1 = 0
                for index_listVec in vec_list:
                    if vector_parallel(index_listVec, vec, 10) == True:
                        count_1 += 1
                if count_1 == 0:
                    vec_list = np.append(vec_list, [vec], axis=0)
                    cv2.circle(ROI, (x0_c, y0_c), 1, (0, 0, 255), 2)
                    cv2.circle(ROI, (x1_c, y1_c), 1, (0, 0, 255), 2)
                    list_all_point = np.append(
                        list_all_point, [[x0_c, y0_c]], axis=0)
                    list_all_point = np.append(
                        list_all_point, [[x1_c, y1_c]], axis=0)

    isCenter = False
    for a_index in range(0, len(list_all_point)):
        for b_index in range(a_index + 1, len(list_all_point)):
            if distanse(list_all_point[a_index][0], list_all_point[a_index][1],
                        list_all_point[b_index][0], list_all_point[b_index][1]) < 5:
                avg_point_x = (
                    list_all_point[a_index][0] + list_all_point[b_index][0]) / 2
                avg_point_y = (
                    list_all_point[a_index][1] + list_all_point[b_index][1]) / 2
                if avg_point_x > leng - 5 and avg_point_x < leng + 5 and avg_point_y > leng - 5 and avg_point_y < leng + 5:
                    isCenter = True
                    break

    if (len(vec_list) == 1 and isCenter == False) or (len(vec_list) == 0 and isCenter == False):
        return ROI, False
    return ROI, True


def vector_normalization(x0, y0, x1, y1):
    x_v = x1 - x0
    y_v = y1 - y0
    if x_v == 0:
        return np.array(([0, 1]), dtype=np.float)
    if y_v == 0:
        return np.array(([1, 0]), dtype=np.float)
    rate = 1 / x_v
    return np.array(([x_v * rate, y_v * rate]), dtype=np.float)


def vector_parallel(vec1, vec2, value_parallel=10):
    # nguoc huongr
    """

    :param vec1: array got 2 value x1,y1
    :param vec2: array got 2 value x2,y2
    :param value_parallel: value show that can parallel
    :return: True is parallel
    False in not parallel
    """
    if vec1[0] == vec2[0] and vec1[1] == vec2[1]:
        return True
    tichVoHuong = vec1[0] * vec1[0] + vec1[1] * vec2[1]
    tichDoDai = math.sqrt(
        pow(vec1[0], 2) + pow(vec1[1], 2)) * math.sqrt(pow(vec2[0], 2) + pow(vec2[1], 2))

    cos_apha = abs(tichVoHuong/tichDoDai)
    if cos_apha > 1:
        cos_apha = cos_apha - 1
    apha = math.acos(cos_apha)

    # chuyen sang radian sang do
    if apha < value_parallel * (math.pi / 180) and apha > 0:
        return True
    return False


def remove_vector_parallel(list_vec, img):
    """
    :param list_vec: check list of vector shape (n,2)
    :param img: y    :return: None but remove all parallel
    """
    for index_vector in range(len(list_vec)):
        count = 0
        for index_search in range(index_vector + 1, len(list_vec)):
            if vector_parallel(list_vec[index_vector], list_vec[index_search]):
                count += 1
        if count > 5:
            pass


def init(folder1, folder2):
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


# def conner(dir_true):
#     dir_conner = "results/conner"
#     dir_line = "results/line"
#     if os.path.exists(dir_line) is False:
#         os.mkdir(dir_line)
#     if os.path.exists(dir_conner) is False:
#         os.mkdir(dir_conner)
#     # dir_true = os.path.join("train", "true")

#     init('./results/conner/', './results/line/')

#     for index_dir in os.listdir(dir_true):
#         img = cv2.imread(os.path.join(dir_true, index_dir))
#         w, h, c = img.shape
#         # img = cv2.resize(img, (int(h/2), int(w/2)),
#         #                  interpolation=cv2.INTER_CUBIC)
#         img2 = copy.copy(img)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#         edges2 = feature.canny(img_gray, sigma=1.25)
#         edges2 = np.asarray(edges2, np.uint8)
#         edges2[edges2 == True] = 255

#         corners = cv2.goodFeaturesToTrack(
#             img_gray, 100, minDistance=20, qualityLevel=0.5)
#         corners = np.int0(corners)
#         list_shiTomasi = np.empty((0, 2), dtype=np.uint16)
#         for i in corners:
#             x, y = i.ravel()
#             list_shiTomasi = np.append(list_shiTomasi, [[x, y]], axis=0)

#         lsd = cv2.createLineSegmentDetector(0)

#         dlines = lsd.detect(img_gray)
#         all_lines = np.empty(shape=(0, 4), dtype=np.uint16)
#         bl_img = np.copy(img_gray) * 0

#         list_endOfLine = np.empty((0, 2), dtype=np.uint16, order='C')
#         list_end = np.empty((0, 2), dtype=np.uint16, order='C')
#         list_start = np.empty((0, 2), dtype=np.uint16, order='C')
#         list_point = np.empty((0, 2), dtype=np.uint16, order='C')
#         list_vector = np.empty((0, 2), dtype=np.float, order='C')
#         for dline in dlines[0]:
#             x0 = int(round(dline[0][0]))
#             y0 = int(round(dline[0][1]))
#             x1 = int(round(dline[0][2]))
#             y1 = int(round(dline[0][3]))

#             if remove_short_line(x0, y0, x1, y1, w, h) is False:
#                 vec = vector_normalization(x0, y0, x1, y1)
#                 dlines = np.append(dlines, [x0, y0, x1, y1], axis=0)

#                 vec = vector_normalization(x0, y0, x1, y1)
#                 list_vector = np.append(list_vector, [vec], axis=0)
#                 list_start = np.append(list_start, [[x0, y0]], axis=0)
#                 list_end = np.append(list_end, [[x1, y1]], axis=0)
#                 list_point = np.append(list_point, [[x0, y0]], axis=0)
#                 list_point = np.append(list_point, [[x1, y1]], axis=0)

#         remove_vector_parallel(list_vector, img)
#         count = 0

#         for index_endLine in list_start:
#             cv2.circle(img2, (index_endLine), 2, (255, 0, 255), 2)
#             ROI1 = conner_findder(img, index_endLine[0], index_endLine[1], 20)
#             if ROI1[1] == True:
#                 cv2.circle(img2, (index_endLine), 2, (0, 255, 255), 2)
#         for index_endLine in list_end:
#             cv2.circle(img2, (index_endLine), 2, (255, 0, 255), 2)

#             ROI1 = conner_findder(img, index_endLine[0], index_endLine[1], 20)
#             if ROI1[1] == True:
#                 cv2.circle(img2, (index_endLine), 2, (0, 255, 255), 2)
#         print(index_dir)
#         cv2.imshow("result_conner", img2)
#         cv2.waitKey()
#         cv2.imwrite(os.path.join(dir_conner, index_dir), img2)
#         print('Find conner done !!!')

# if __name__ == "__main__":

#     dir_true = "test"
#     conner(dir_true)
