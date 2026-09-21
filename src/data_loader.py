"""
BC카드 소비데이터 로딩 모듈

Step 1의 데이터 적재를 담당한다. 공식 안내문서(ABP_CONTEST_DATA 설명)와
실측 데이터 사이에 존재하는 두 가지 불일치(컬럼명 대소문자, 외국인 AGE_CD)를
로딩 시점에 검증(assert)하여, 이후 단계에서 조용히 잘못된 가정 위에서
분석이 진행되는 것을 막는다.
"""
from pathlib import Path

import pandas as pd

# src/ 의 부모 = 프로젝트 루트. cwd(작업 디렉터리)에 의존하지 않도록 절대경로로 고정한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "ABP_CONTEST_DATA.csv"

# 명세서(§2-1)에 정의된 기대 스키마
EXPECTED_COLUMNS = [
    "STRD_YYMM",
    "SIDO_NM",
    "CCG_NM",
    "GENDER_CD",
    "AGE_CD",
    "TP_BUZ_NO",
    "TP_BUZ_NM",
    "amt",
    "cnt",
]

# 데이터에 실제로 존재하는 업종 11종 (공백 정규화 이후 기준)
EXPECTED_BUZ_NAMES = {
    "서양음식", "일반한식", "중국음식", "일식회집", "한정식",
    "갈비전문점", "편의점", "대형할인점", "슈퍼마켓", "제과점", "스넥",
}

EXPECTED_YYMM = {"202601", "202602", "202603", "202604", "202605", "202606"}


def load_raw_data(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """BC카드 원본 CSV를 로드하고 스키마를 검증한다.

    - 인코딩은 utf-8-sig 고정 (BOM 포함 파일이므로 일반 utf-8로 읽으면
      첫 컬럼명 앞에 BOM 문자가 남아 'STRD_YYMM' 매칭이 깨진다).
    - GENDER_CD/AGE_CD는 '01', '1' 등 표기가 섞일 수 있어 문자열로 강제하고 좌우 공백을 제거한다.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"원본 데이터 파일을 찾을 수 없습니다: {path}\n"
            "data/raw/ 에 ABP_CONTEST_DATA.csv 를 위치시키세요 (git에는 포함되지 않음)."
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"GENDER_CD": str, "AGE_CD": str, "TP_BUZ_NM": str},
    )

    # --- 컬럼 스키마 검증 (실측 함정 #1: amt/cnt가 소문자로 제공됨) ---
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    assert not missing, (
        f"기대한 컬럼이 없습니다: {missing}. "
        f"실제 컬럼: {df.columns.tolist()} "
        "(공식 안내문서는 AMT/CNT 대문자를 명시하지만 실측 파일은 소문자 amt/cnt 이므로 주의)"
    )

    df["GENDER_CD"] = df["GENDER_CD"].str.strip()
    df["AGE_CD"] = df["AGE_CD"].str.strip()
    df["STRD_YYMM"] = df["STRD_YYMM"].astype(str).str.strip()

    # --- 값 범위 검증 ---
    assert set(df["STRD_YYMM"].unique()) <= EXPECTED_YYMM, (
        f"예상 밖의 기준년월 값 발견: {set(df['STRD_YYMM'].unique()) - EXPECTED_YYMM}"
    )
    assert set(df["GENDER_CD"].unique()) <= {"1", "2", "3", "x"}, (
        f"예상 밖의 GENDER_CD 값 발견: {set(df['GENDER_CD'].unique()) - {'1', '2', '3', 'x'}}"
    )

    # --- 실측 함정 #2: 외국인(GENDER_CD='3')도 AGE_CD 1~6이 채워져 있음을 확인 ---
    foreign_age_codes = set(df.loc[df["GENDER_CD"] == "3", "AGE_CD"].unique())
    assert foreign_age_codes <= {"1", "2", "3", "4", "5", "6"}, (
        f"외국인 AGE_CD에 예상 밖 값: {foreign_age_codes}"
    )
    assert "x" not in foreign_age_codes, (
        "외국인(GENDER_CD=3)의 AGE_CD에 'x'가 없어야 한다는 가정이 깨졌습니다. "
        "공식 안내문서가 맞을 수 있으니 재검증 필요."
    )

    return df


def validate_business_categories(df: pd.DataFrame, buz_col: str = "TP_BUZ_NM") -> None:
    """업종명이 공백 정규화 후 정확히 11종으로 떨어지는지 검증한다 (실측 함정 #3)."""
    normalized = df[buz_col].str.replace(" ", "", regex=False)
    actual = set(normalized.unique())
    assert actual == EXPECTED_BUZ_NAMES, (
        f"업종 정규화 후에도 기대한 11종과 다릅니다.\n"
        f"기대: {EXPECTED_BUZ_NAMES}\n실제: {actual}"
    )


if __name__ == "__main__":
    raw = load_raw_data()
    validate_business_categories(raw)
    print(f"[OK] 로딩 완료: {len(raw):,}행, 컬럼: {raw.columns.tolist()}")
    print(f"[OK] 기준년월 범위: {sorted(raw['STRD_YYMM'].unique())}")
    print(f"[OK] 성별 분포:\n{raw['GENDER_CD'].value_counts()}")
