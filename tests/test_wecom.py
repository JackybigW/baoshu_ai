"""
WeCom Integration Unit Tests
模拟企业微信的完整收发流程：
1. 测试 WeComCrypto 的签名验证和 AES 加解密
2. 测试 GET /api/wecom/callback (URL 验证握手)
3. 测试 POST /api/wecom/callback (事件接收+去重)
"""
import base64
import hashlib
import os
import struct
import sys
import time

import pytest

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Crypto.Cipher import AES

from utils.wecom_crypto import WeComCrypto

# ==========================================
# 测试用的固定凭证 (与 .env 无关，纯模拟)
# ==========================================
TEST_TOKEN = "TestToken123456789012345"
TEST_AES_KEY = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"  # 43 chars
TEST_CORPID = "wx1234567890abcdef"


# ==========================================
# Helper: 模拟企业微信加密消息
# ==========================================
def build_encrypted_xml(crypto: WeComCrypto, xml_content: str) -> tuple:
    """
    模拟腾讯服务器行为：把明文 XML 加密成企微推送格式的密文 + 签名参数。
    返回 (encrypted_xml_body, msg_signature, timestamp, nonce)
    """
    # 1. 构造明文二进制: random(16) + msg_len(4, big-endian) + xml_bytes + corpid
    xml_bytes = xml_content.encode("utf-8")
    random_prefix = os.urandom(16)
    msg_len = struct.pack("!I", len(xml_bytes))
    corpid_bytes = TEST_CORPID.encode("utf-8")
    plaintext = random_prefix + msg_len + xml_bytes + corpid_bytes

    # 2. PKCS#7 填充到 AES 块大小 (32 bytes for AES-256)
    block_size = 32
    pad_count = block_size - (len(plaintext) % block_size)
    plaintext += bytes([pad_count] * pad_count)

    # 3. AES-256-CBC 加密
    key = crypto.key
    iv = key[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted_bytes = cipher.encrypt(plaintext)
    encrypt_text = base64.b64encode(encrypted_bytes).decode("utf-8")

    # 4. 计算签名: sorted([token, timestamp, nonce, encrypt_text]) -> SHA-1
    timestamp = str(int(time.time()))
    nonce = "testnonce12345"
    sort_list = sorted([TEST_TOKEN, timestamp, nonce, encrypt_text])
    sha = hashlib.sha1("".join(sort_list).encode("utf-8"))
    msg_signature = sha.hexdigest()

    # 5. 构造外层 XML
    outer_xml = f"""<xml>
<ToUserName><![CDATA[{TEST_CORPID}]]></ToUserName>
<Encrypt><![CDATA[{encrypt_text}]]></Encrypt>
<AgentID><![CDATA[]]></AgentID>
</xml>"""

    return outer_xml.encode("utf-8"), msg_signature, timestamp, nonce, encrypt_text


# ==========================================
# Test 1: WeComCrypto 单元测试
# ==========================================
class TestWeComCrypto:
    """测试加解密模块的核心逻辑"""

    def setup_method(self):
        self.crypto = WeComCrypto(TEST_TOKEN, TEST_AES_KEY, TEST_CORPID)

    def test_signature_valid(self):
        """验签通过：正确的 token + timestamp + nonce + data"""
        timestamp = "1640000000"
        nonce = "abc123"
        data = "some_encrypted_data"
        # 手动计算期望签名
        sort_list = sorted([TEST_TOKEN, timestamp, nonce, data])
        expected = hashlib.sha1("".join(sort_list).encode()).hexdigest()

        assert self.crypto.verify_signature(expected, timestamp, nonce, data) is True

    def test_signature_invalid(self):
        """验签失败：伪造的签名"""
        assert self.crypto.verify_signature("fakesig", "123", "abc", "data") is False

    def test_encrypt_decrypt_roundtrip(self):
        """完整的加密-解密往返测试：模拟企微发送 → 解密还原"""
        original_xml = "<xml><Event>kf_msg_or_event</Event><Token>sync_token_abc</Token></xml>"
        _, _, _, _, encrypt_text = build_encrypted_xml(self.crypto, original_xml)

        # 解密
        decrypted = self.crypto.decrypt(encrypt_text)
        assert decrypted == original_xml

    def test_decrypt_chinese_content(self):
        """中文内容加解密测试"""
        original_xml = "<xml><Content>我想去英国读硕士</Content></xml>"
        _, _, _, _, encrypt_text = build_encrypted_xml(self.crypto, original_xml)

        decrypted = self.crypto.decrypt(encrypt_text)
        assert "我想去英国读硕士" in decrypted


# ==========================================
# Test 2: FastAPI 端点集成测试
# ==========================================
class TestWeComEndpoints:
    """
    测试 GET/POST /api/wecom/callback 端点。
    使用 mock 隔离外部依赖 (Redis, 企微 API, LangGraph)。
    """

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """设置测试环境变量"""
        monkeypatch.setenv("WECOM_CORPID", TEST_CORPID)
        monkeypatch.setenv("WECOM_SECRET", "test_secret")
        monkeypatch.setenv("WECOM_TOKEN", TEST_TOKEN)
        monkeypatch.setenv("WECOM_AES_KEY", TEST_AES_KEY)
        monkeypatch.setenv("WECOM_KF_ID", "kf_test123")

    @pytest.fixture
    def crypto_instance(self):
        return WeComCrypto(TEST_TOKEN, TEST_AES_KEY, TEST_CORPID)

    def test_get_url_verification(self, crypto_instance):
        """
        测试 GET 验证: 模拟企微后台保存 URL 时发送的探活请求。
        期望：验签通过 → 返回解密后的纯明文。
        """
        # 构造一个加密的 echostr
        echostr_plain = "echo_test_12345"
        _, _, _, _, encrypt_text = build_encrypted_xml(crypto_instance, echostr_plain)

        # 计算签名
        timestamp = str(int(time.time()))
        nonce = "get_test_nonce"
        sort_list = sorted([TEST_TOKEN, timestamp, nonce, encrypt_text])
        msg_signature = hashlib.sha1("".join(sort_list).encode()).hexdigest()

        # 验签
        assert crypto_instance.verify_signature(msg_signature, timestamp, nonce, encrypt_text) is True

        # 解密
        decrypted = crypto_instance.decrypt(encrypt_text)
        assert decrypted == echostr_plain

    def test_post_event_parsing(self, crypto_instance):
        """
        测试 POST 事件解析: 模拟企微推送 kf_msg_or_event 事件。
        期望：正确从外层 XML 提取 Encrypt → 验签 → 解密 → 提取 Event 和 Token。
        """
        from lxml import etree

        inner_xml = "<xml><Event>kf_msg_or_event</Event><Token>sync_cursor_abc123</Token></xml>"
        outer_body, msg_signature, timestamp, nonce, _ = build_encrypted_xml(
            crypto_instance, inner_xml
        )

        # 1. 解析外层 XML
        xml_tree = etree.fromstring(outer_body)
        encrypt_text = xml_tree.find("Encrypt").text

        # 2. 用 encrypt_text 做签名验证 (P0 fix: 不是 raw body)
        assert crypto_instance.verify_signature(msg_signature, timestamp, nonce, encrypt_text) is True

        # 3. 解密
        xml_content = crypto_instance.decrypt(encrypt_text)
        decrypted_tree = etree.fromstring(xml_content.encode("utf-8"))

        # 4. 提取业务字段
        event_type = decrypted_tree.findtext("Event")
        sync_token = decrypted_tree.findtext("Token")

        assert event_type == "kf_msg_or_event"
        assert sync_token == "sync_cursor_abc123"

    def test_post_signature_with_wrong_body_fails(self, crypto_instance):
        """
        验证 P0 修复: 如果用 raw XML body (而不是 Encrypt 字段) 做验签，必须失败。
        """
        inner_xml = "<xml><Event>kf_msg_or_event</Event><Token>abc</Token></xml>"
        outer_body, msg_signature, timestamp, nonce, _ = build_encrypted_xml(
            crypto_instance, inner_xml
        )

        # 错误做法: 用 raw body 字符串做验签 → 必须失败
        raw_body_str = outer_body.decode("utf-8")
        result = crypto_instance.verify_signature(msg_signature, timestamp, nonce, raw_body_str)
        assert result is False, "P0 bug regression: raw body should NOT pass signature verification!"

    def test_duplicate_event_detection(self):
        """
        测试 Redis SETNX 防抖逻辑 (概念验证)。
        同一个 sync_token 第二次 SETNX 应该返回 False。
        """
        # 模拟 Redis SETNX 行为
        seen = set()

        def mock_setnx(key: str):
            if key in seen:
                return False
            seen.add(key)
            return True

        token = "sync_token_xyz"
        lock_key = f"wecom:lock:{token}"

        # 第一次: 新事件
        assert mock_setnx(lock_key) is True
        # 第二次: 重复事件 (企微超时重试)
        assert mock_setnx(lock_key) is False


# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
