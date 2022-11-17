# -*- coding = utf-8 -*-
# 代码作者: 染瑾年ii
# 创建时间: 2022/11/17 13:43 
# 文件名称: 1.py
# 编程软件: PyCharm

import numpy as np
import cv2 as cv
import  time
# 加载彩色灰度图像
img = cv.imread('tupian.jpg',0)
# **cv.imshow()**在窗口中显示图像。窗口自动适合图像尺寸。第一个参数是窗口名称，它是一个字符串。第二个参数是我们的对象。你可以根据需要创建任意多个窗口，但可以使用不同的窗口名称。
cv.imshow('image',img)
# cv.waitKey()是一个键盘绑定函数。其参数是以毫秒为单位的时间。该函数等待任何键盘事件指定的毫秒。如果您在这段时间内按下任何键，程序将继续运行。如果**0**被传递，它将无限期地等待一次敲击键。它也可以设置为检测特定的按键，
k = cv.waitKey(0)
print("k的值",k)
print("ord函数S的值",ord('S'))
if k == 27:         # 等待ESC退出
    print("走Esc了")
    cv.destroyAllWindows()
elif k == ord('S'): # 等待关键字，保存和退出
    print("走按键了")
    cv.imwrite('tupian_grey.jpg',img)
    cv.destroyAllWindows()
time.sleep(10)


# retval, dst = cv.threshold(img, 10, 100, cv.THRESH_BINARY)
# print("图像",img)
# print("dst修改后的图像",dst)
# print("retval是什么",retval)