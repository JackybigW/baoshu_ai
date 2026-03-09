import base64
import hashlib
import struct
from Crypto.Cipher import AES

class WeComCrypto:
    def __init__(self, token: str, encoding_aes_key: str, corpid: str):
        self.token = token
        self.corpid = corpid
        # The true 32-byte AES key requires appending '=' to the 43-char config string
        self.key = base64.b64decode(encoding_aes_key + "=")

    def verify_signature(self, msg_signature: str, timestamp: str, nonce: str, data: str) -> bool:
        """
        Verifies the SHA-1 signature of the incoming HTTP request.
        """
        sort_list = sorted([self.token, timestamp, nonce, data])
        sort_str = "".join(sort_list)
        sha = hashlib.sha1()
        sha.update(sort_str.encode("utf-8"))
        return sha.hexdigest() == msg_signature

    def decrypt(self, encrypted_text: str) -> str:
        """
        Decrypts the AES-256-CBC encrypted XML payload.
        """
        cryptor = AES.new(self.key, AES.MODE_CBC, self.key[:16])
        decrypted_bytes = cryptor.decrypt(base64.b64decode(encrypted_text))
        
        # Remove PKCS#7 padding
        pad = decrypted_bytes[-1]
        decrypted_bytes = decrypted_bytes[:-pad]
        
        # Discard 16-bye random prefix, then read 4-byte network-order integer for XML length
        msg_len = struct.unpack("!I", decrypted_bytes[16:20])[0]
        # Extract the pure XML payload based on the parsed length
        xml_content = decrypted_bytes[20:20+msg_len].decode("utf-8")
        return xml_content
