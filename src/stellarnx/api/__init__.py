"""HTTP 边界层。只做参数校验与序列化，业务逻辑一律下沉到 pipeline。

作者: 晨星
"""

from stellarnx.api.app import create_app

__all__ = ["create_app"]
