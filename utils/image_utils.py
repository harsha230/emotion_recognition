import base64

def read_image_bytes(path: str):
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        raise e


def encode_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")
