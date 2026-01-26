from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid
from py_vapid.utils import b64urlencode


def main() -> None:
    vapid = Vapid()
    vapid.generate_keys()
    public_key = b64urlencode(
        vapid.public_key.public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )
    )
    private_value = vapid.private_key.private_numbers().private_value
    private_key = b64urlencode(private_value.to_bytes(32, "big"))
    print("VAPID_PUBLIC_KEY=" + public_key)
    print("VAPID_PRIVATE_KEY=" + private_key)


if __name__ == "__main__":
    main()
