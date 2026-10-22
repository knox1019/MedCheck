"""包标记文件。

Python 靠它判断这个目录是一个「包」，这样才能用 from app.xxx import yyy
的方式导入里面的模块。内容为空是正常的，不要删。
单元测试。文件名以 test_ 开头，函数名也要以 test_ 开头，否则 pytest 不会收集。
"""
