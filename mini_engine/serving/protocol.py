"""Validation and errors for the supported greedy chat protocol."""


class APIError(Exception):
    def __init__(self, status, message, *, code="invalid_request", param=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param

    def payload(self):
        return {"error": {"message": self.message, "type": "server_error" if self.status >= 500
                          else "invalid_request_error", "param": self.param, "code": self.code}}


def validate_chat_request(payload, *, model_name, max_new_tokens):
    """Validate payload without touching a tokenizer, scheduler, or model."""
    if not isinstance(payload, dict):
        raise APIError(400, "JSON body must be an object")
    allowed = {"model", "messages", "max_tokens", "stream", "temperature", "stream_options"}
    if set(payload) - allowed:
        raise APIError(400, "Unsupported fields: " + ", ".join(sorted(set(payload) - allowed)))
    if payload.get("model") != model_name:
        raise APIError(404, "Unknown model", code="model_not_found", param="model")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise APIError(400, "messages must contain 1 to 128 text messages", param="messages")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or not isinstance(message["content"], str) or not message["content"]):
            raise APIError(400, "Each message needs a supported role and nonempty string content", param="messages")
    tokens = payload.get("max_tokens", min(128, max_new_tokens))
    if type(tokens) is not int or not 1 <= tokens <= max_new_tokens:
        raise APIError(400, f"max_tokens must be an integer in [1, {max_new_tokens}]", param="max_tokens")
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise APIError(400, "stream must be a boolean", param="stream")
    stream_options = payload.get("stream_options", {})
    if (not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}
            or type(stream_options.get("include_usage", False)) is not bool
            or ("stream_options" in payload and not stream)):
        raise APIError(400, "stream_options only supports include_usage:boolean with stream=true", param="stream_options")
    temperature = payload.get("temperature", 0)
    if type(temperature) not in (int, float) or temperature != 0:
        raise APIError(400, "Only greedy decoding is supported; temperature must be 0", param="temperature")
    return [dict(message) for message in messages], tokens, stream, stream_options.get("include_usage", False)
