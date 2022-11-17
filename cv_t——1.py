import cv2
import numpy as np
import imutils

def cv2gray(file_path, fin_file_path):
    """
    图片灰度处理
    :param file_path:
    :param fin_file_path:
    :return:
    """
    img = cv2.imread(file_path)
    new = np.clip(1.4057577998008846*img-38.33089999653017, 0, 255).astype(np.uint8)
    img2 = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    # 最大类间方差法(大津算法)，thresh会被忽略，自动计算一个阈值
    retval, dst = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(fin_file_path, dst)



def template_matching(image_full_path, tmpl_full_path, fin_full_path, min_confidence=0.93, ):
    """
    :param image_full_path: 输入图片路径，在该图片中查找模板图像
    :param tmpl_full_path: 模板图片，在输入图片中查找此模板
    :param min_confidence: 最小信心度，该方法返回大于此信心度的结果
    :return: 返回result_dic ==> {信心度：（坐标Tuple）}, sorted(confidence_list,reverse=True)==>信心度降序排列列表
    """
    img = cv2.imread(image_full_path)  # 读取输入图片
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图，采用CV_BGR2GRAY,转换公式Gray = 0.1140*B + 0.5870*G + 0.2989*R
    # template = cv2.imread(tmpl_full_path, cv2.IMREAD_GRAYSCALE)  # 读取模板图片的灰度图
    template = cv2.imread(tmpl_full_path)  # 为保证输入图片与模板一致性，两张图片用相同方法读取灰度图片
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]     # 获取模板的宽和高(绘图时使用，本函数内无意义）
    w1, h1 = gray_img.shape[::-1]     # 获取模板的宽和高(绘图时使用，本函数内无意义）
    h2, w2 = template.shape[:2]

    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)  # TM_CCOEFF_NORMED 标准相关匹配
    loc = np.where(res >= min_confidence)   # 在结果中筛选大于最小信心度的结果
    result_dic = {}
    confidence_list = []

    for pt in zip(*loc[::-1]):
        result_dic[res[pt[1]][pt[0]]] = pt
        confidence_list.append(res[pt[1]][pt[0]])
        # print(pt[1],pt[1]+h, pt[0],pt[0]+w)

        img[pt[1]-h:pt[1]+h, pt[0]:pt[0]+w] = 255
        # cv2.floodFill(img, mask, (pt[1], pt[0]), (0, 0, 255), cv2.FLOODFILL_MASK_ONLY)
        # cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)  # 在目标位置绘制一个红色矩形框，边宽3px
    # cv2.imshow("fill", img)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel)
    # dst = cv2.dilate(img, kernel,iterations=1)
    cv2.imwrite(fin_full_path, erosion)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result_dic, sorted(confidence_list, reverse=True)


def diff_size_template_matching(image_full_path, tmpl_full_path, min_confidence=0.95, ):
    """
    :param image_full_path: 输入图片路径，在该图片中查找模板图像
    :param tmpl_full_path: 模板图片，在输入图片中查找此模板
    :param min_confidence:  最小信心度，该方法返回大于此信心度的结果
    :return: 返回result_dic ==> {信心度：（坐标Tuple）}, sorted(confidence_list,reverse=True)==>信心度降序排列列表
    """
    template = cv2.imread(tmpl_full_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # 获得灰度模板

    image = cv2.imread(image_full_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 获得灰度图片
    result_dic = {}
    confidence_list = []
    for scale in np.linspace(0.5, 1.0, 1000)[::-1]:  # 1倍到0.2倍 分100级变化(级数越高，可能获得的匹配精度越高，处理越慢)
        resized = imutils.resize(template, width=int(template.shape[1] * scale))  # 以scale 为倍数改变模板大小
        (w, h) = resized.shape[::-1]    # 求改变后的模板宽高
        result = cv2.matchTemplate(gray_img, resized, cv2.TM_CCOEFF_NORMED)    # 每次改变都进行一次匹配
        loc = np.where(result >= min_confidence)  # 在结果中筛选大于最小信心度的结果
        for pt in zip(*loc[::-1]):
            result_dic[result[pt[1]][pt[0]]] = pt
            confidence_list.append(result[pt[1]][pt[0]])
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
        cv2.imwrite('/Users/woobrain/Downloads/10/img_1.jpg', image)

    return result_dic, sorted(confidence_list, reverse=True)


if __name__ == '__main__':
    cv2gray("/Users/woobrain/Downloads/10/10.jpg",'/Users/woobrain/Downloads/10/removed.jpg')
    result_dic,a = template_matching("/Users/woobrain/Downloads/10/10.jpg", "/Users/woobrain/Downloads/10/tmp2.jpg",
                                     '/Users/woobrain/Downloads/10/img_1.jpg', min_confidence=0.85, )
    # print(a)