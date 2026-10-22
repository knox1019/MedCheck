"""health 接口的单元测试。

用 TestClient 有一个关键好处：它不会真的启动服务器。
请求直接在同一个进程里交给 FastAPI 处理，所以跑得很快，
也不用管端口占用、不用先开 uvicorn。
"""

from fastapi.testclient import TestClient

from app.main import app

# 模块级创建一次，所有用例复用。TestClient 内部不会占用网络端口。
client = TestClient(app)


def test_health_returns_200():
    """验证接口能正常响应。

    函数名必须以 test_ 开头，否则 pytest 不会收集这个用例。
    这是约定，不是语法要求。
    """
    response = client.get("/health")
    assert response.status_code == 200


def test_health_has_required_fields():
    """验证返回内容里包含必要字段。

    注意这里只断言「字段存在」，没有断言「值正确」——
    这属于较弱的断言：字段存在但值错了，测试照样通过。
    更严的写法是 assert data["env"] == "dev"，但那样测试会依赖运行环境
    （换到生产环境就会失败）。取舍取决于你希望这个测试守什么。
    """
    # .json() 把响应体从 JSON 字符串转成 Python 字典
    data = client.get("/health").json()

    assert data["status"] == "ok"
    assert "env" in data
    assert "version" in data
