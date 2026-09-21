"""
카드데이터 행정구역명 <-> 법무부 등록외국인 통계 행정구역명 매핑 테이블

실측 결과 (2026-09-21 검증):
  - 카드데이터 고유 (SIDO_NM, CCG_NM) 조합: 255개
  - 그 중 법무부 자료의 (시도, 시군구) 표기와 "완전히 동일한 문자열"로 매칭되는 것: 254개 (99.6%)
  - 유일한 예외: 세종특별자치시. 카드데이터는 SIDO_NM=CCG_NM='세종특별자치시'로 표기하지만
    법무부 자료는 세종을 시군구 하위 구분 없이 시도 레벨 '총계' 행(시군구 컬럼이 빈 문자열)으로만 제공한다.
  - 매칭 실패율이 0.4%로 §9-2의 중단 기준(20% 초과)에 한참 못 미치므로 자동 진행한다.

이 모듈은 (SIDO_NM, CCG_NM) 문자열이 기본적으로 그대로 일치한다는 전제 위에,
예외 케이스만 명시적으로 override 하는 방식을 쓴다. 향후 법무부 자료 회차가 바뀌어
새로운 표기 불일치가 발견되면 EXPLICIT_OVERRIDES 에 추가하면 된다.
"""
from typing import Optional

# 카드데이터 (SIDO_NM, CCG_NM) -> 법무부 자료 (시도, 시군구) 명시적 예외 매핑
# 법무부 자료에서 세종은 시군구 세분류가 없어 시군구 컬럼이 빈 문자열('')로 표기됨
EXPLICIT_OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {
    ("세종특별자치시", "세종특별자치시"): ("세종특별자치시", ""),
}


def map_card_region_to_moj(sido: str, ccg: str) -> tuple[str, str]:
    """카드데이터의 (SIDO_NM, CCG_NM)을 법무부 자료의 (시도, 시군구) 키로 변환한다."""
    return EXPLICIT_OVERRIDES.get((sido, ccg), (sido, ccg))


def build_join_report(card_pairs: set, moj_pairs: set) -> dict:
    """카드데이터 지역 목록과 법무부 지역 목록 간 조인 성공/실패 현황을 계산한다."""
    mapped = {map_card_region_to_moj(s, c) for s, c in card_pairs}
    matched = mapped & moj_pairs
    unmatched = mapped - moj_pairs
    fail_rate = len(unmatched) / len(mapped) if mapped else 0.0
    return {
        "total": len(mapped),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "unmatched_list": sorted(unmatched),
        "fail_rate": fail_rate,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from external_data import load_moj_visa_by_region

    card = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "ABP_CONTEST_DATA.csv", encoding="utf-8-sig", dtype=str)
    card_pairs = set(zip(card["SIDO_NM"], card["CCG_NM"]))

    moj = load_moj_visa_by_region()
    moj_pairs = set(zip(moj["시도"], moj["시군구"]))

    report = build_join_report(card_pairs, moj_pairs)
    print(f"[OK] 조인 대상 지역 수: {report['total']}")
    print(f"[OK] 매칭 성공: {report['matched']} ({(1 - report['fail_rate']) * 100:.1f}%)")
    print(f"[OK] 매칭 실패: {report['unmatched']} ({report['fail_rate'] * 100:.1f}%) -> {report['unmatched_list']}")
