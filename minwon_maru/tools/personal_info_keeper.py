import re
from copy import deepcopy
from typing import Tuple, Dict, Any, Union

################################################################################
# 마스킹 핸들러
################################################################################
def mask_phone_simple(match: re.Match) -> str:
    # 전화번호는 "간단/직관"하게: 모든 숫자를 * 로, 구분자는 유지
    s = match.group(0)
    return re.sub(r"\d", "*", s)

def mask_rrn(match: re.Match) -> str:
    left = match.group("left")
    return f"{left}-*******"

def mask_email(match: re.Match) -> str:
    local = match.group("local")
    domain = match.group("domain")
    if len(local) <= 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[0] + "*"*(len(local)-2) + local[-1]
    return f"{masked_local}@{domain}"

def mask_card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    masked = "*" * 12 + digits[-4:]
    return f"{masked[0:4]}-{masked[4:8]}-{masked[8:12]}-{masked[12:16]}"

def mask_passport(match: re.Match) -> str:
    s = match.group(0)
    return s[0] + "*"*8


################################################################################
# 패턴 (카드 먼저 → 전화)
################################################################################
# 주민등록번호
rrn_hyphen  = re.compile(r"(?P<left>\d{6})-(?P<right>\d{7})")
rrn_compact = re.compile(r"(?P<left>\d{6})(?P<right>\d{7})(?!\d)")

# 이메일
email_pattern = re.compile(r"(?P<local>[A-Za-z0-9._%+\-]+)@(?P<domain>[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")

# 신용카드: 4-4-4-4 또는 16자리 연속
card_grouped = re.compile(r"(?:\d{4}[-\s]?){3}\d{4}")
card_compact = re.compile(r"\d{16}")

# 여권(대한민국): 영문 1자 + 숫자 8
passport_pattern = re.compile(r"\b([A-Za-z])\d{8}\b")

# 전화번호
# - 한국형(0으로 시작): 02/0xx/0xxx + 구분자 + 3~4 + 구분자 + 4
phone_kr = re.compile(r"(?:0\d{1,2})[-.\s]?\d{3,4}[-.\s]?\d{4}")
# - 범용(앞자리가 0 아닐 수도): 2~4 + 구분자 + 3~4 + 구분자 + 4
#   뒤에 4자리 그룹이 더 이어져(카드처럼) 보이지 않도록 차단
phone_generic = re.compile(r"\d{2,4}[-.\s]\d{3,4}[-.\s]\d{4}(?!\s*[-.\s]?\d{4})")
# - 붙임표 없는 10~11자리(국내 휴대/지역 합산 길이) — 양옆 숫자 이어붙은 16자리 카드 방지
phone_compact_kr = re.compile(r"(?<!\d)\d{10,11}(?!\d)")

################################################################################
# 핵심 마스킹 함수
################################################################################
def _mask_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # 적용 순서: 더 특수/위험한 것 → 범용
    text = rrn_hyphen.sub(mask_rrn, text)
    text = rrn_compact.sub(mask_rrn, text)
    text = email_pattern.sub(mask_email, text)
    text = card_grouped.sub(mask_card, text)
    text = card_compact.sub(mask_card, text)
    text = passport_pattern.sub(mask_passport, text)

    # 전화(한국형 → 범용 → 붙임표 없는 형태)
    text = phone_kr.sub(mask_phone_simple, text)
    text = phone_generic.sub(mask_phone_simple, text)
    # 붙임표 없는 10~11자리: 0000000000 → **********
    text = phone_compact_kr.sub(lambda m: "*" * len(m.group(0)), text)
    return text

def _mask_any(x):
    if isinstance(x, str):
        return _mask_text(x)
    if isinstance(x, list):
        return [_mask_any(i) for i in x]
    if isinstance(x, dict):
        return {k: _mask_any(v) for k, v in x.items()}
    return x

################################################################################
# 공개 API: 문자열 또는 딕셔너리 둘 다 지원
################################################################################
def personal_info_keeper(inputs: Union[str, Dict[str, Any]]) -> Tuple[Union[str, Dict[str, Any]], Union[str, Dict[str, Any]]]:
    """
    문자열 또는 dict를 받아 마스킹 결과와 원본을 튜플로 반환.
      - str 입력: (masked_str, raw_str)
      - dict 입력: (masked_dict, raw_dict)
    """
    raw_inputs = deepcopy(inputs)
    print("raw_inputs : ", raw_inputs)
    masked = _mask_any(inputs)
    print("masked : ", masked)
    return masked, raw_inputs
