"""统一错误体系。分层：配置错误 / 依赖不可用 / 上游超时 / 校验不通过。

作者: 晨星
"""


class StellarNexusError(Exception):
    """所有本项目异常的基类。"""

    code = "internal_error"
    http_status = 500


class ConfigError(StellarNexusError):
    """配置缺失或非法。"""

    code = "config_error"
    http_status = 500


class DependencyUnavailable(StellarNexusError):
    """外部依赖（Ollama / ONNX 模型文件 / 向量库）不可用。

    这是可降级的关键点：调用方捕获后应切到零依赖实现，而不是让整个请求失败。
    """

    code = "dependency_unavailable"
    http_status = 503


class UpstreamTimeout(StellarNexusError):
    """上游 LLM / 嵌入服务超时。"""

    code = "upstream_timeout"
    http_status = 504


class VerificationFailed(StellarNexusError):
    """答案未通过可证性校验。"""

    code = "verification_failed"
    http_status = 422


class NotFound(StellarNexusError):
    """资源不存在。"""

    code = "not_found"
    http_status = 404
