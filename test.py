# -*- coding = utf-8 -*-
# 代码作者: 染瑾年ii
# 创建时间: 2022/11/17 15:00 
# 文件名称: test.py
# 编程软件: PyCharm
import  psutil
def get_cpu_info():
    cpu_count = psutil.cpu_count(logical=False)  #1代表单核CPU，2代表双核CPU
    xc_count = psutil.cpu_count()                #线程数，如双核四线程
    cpu_percent = round((psutil.cpu_percent(1)), 2)  # cpu使用率
    cpu_info = (cpu_count,xc_count,cpu_percent)
    return cpu_info

print("CPU信息",get_cpu_info())