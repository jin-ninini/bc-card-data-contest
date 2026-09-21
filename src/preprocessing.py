"""
Step 1: 전처리 모듈

data_loader.load_raw_data() 로 불러온 원본 데이터를 정제하여
외국인(GENDER_CD='3') 소비 데이터만 추출한 clean 데이터셋을 만든다.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "foreign_consumption_clean.csv"

# AGE_CD -> 한글 라벨 매핑 (안내문서 §2-1 기준)
AGE_LABELS = {
    "1": "20대 이하",
    "2": "20대",
    "3": "30대",
    "4": "40대",
    "5": "50대",
    "6": "60대 이상",
}

# 업종명 정규화 매핑: 원본에 섞여 있는 공백 변형을 표준 11종 명칭으로 통일
BUZ_NAME_CANONICAL = {
    "서양음식": "서양음식", "일반한식": "일반한식", "중국음식": "중국음식",
    "일식회집": "일식회집", "한정식": "한정식", "갈비전문점": "갈비전문점",
    "편의점": "편의점", "편 의 점": "편의점",
    "대형할인점": "대형할인점",
    "슈퍼마켓": "슈퍼마켓", "슈퍼 마켓": "슈퍼마켓",
    "제과점": "제과점", "제 과 점": "제과점",
    "스넥": "스넥",
}


def normalize_business_name(series: pd.Series) -> pd.Series:
    """업종명의 불규칙한 공백을 제거하여 표준 11종 명칭으로 통일한다."""
    stripped = series.str.replace(" ", "", regex=False)
    return stripped.map(BUZ_NAME_CANONICAL).fillna(stripped)


def clean_and_filter_foreign(df: pd.DataFrame) -> pd.DataFrame:
    """원본 데이터를 정제하고 외국인(GENDER_CD='3') 데이터만 추출한다.

    결측/0 이하 금액·건수 처리 방침:
      data_loader 검증 결과 amt/cnt 모두 결측 0건, 0 이하 값 0건으로 확인됨
      (242,574행 전수 검사). 따라서 별도의 결측치 대체나 행 제거 로직은
      추가하지 않는다 — 데이터가 이미 "집계값이 1건 이상 존재하는 조합만
      수록"된 형태이기 때문으로 추정된다. 향후 데이터 갱신 시 이 가정이
      깨질 수 있으므로 assert로 명시적으로 고정해 둔다.
    """
    assert df["amt"].isnull().sum() == 0 and (df["amt"] <= 0).sum() == 0, (
        "amt에 결측 또는 0 이하 값이 발견되었습니다. 전처리 로직 재검토 필요."
    )
    assert df["cnt"].isnull().sum() == 0 and (df["cnt"] <= 0).sum() == 0, (
        "cnt에 결측 또는 0 이하 값이 발견되었습니다. 전처리 로직 재검토 필요."
    )

    df = df.copy()
    df["TP_BUZ_NM"] = normalize_business_name(df["TP_BUZ_NM"])

    foreign_df = df[df["GENDER_CD"] == "3"].copy()
    foreign_df["AGE_LABEL"] = foreign_df["AGE_CD"].map(AGE_LABELS)
    assert foreign_df["AGE_LABEL"].isnull().sum() == 0, (
        "외국인 데이터 중 AGE_CD -> 한글 라벨 매핑에 실패한 행이 있습니다."
    )

    # 건당 평균 결제금액은 이후 여러 단계(피처 엔지니어링·세그먼트 분석)에서
    # 반복 사용되므로 clean 데이터 시점에 미리 계산해 둔다.
    foreign_df["amt_per_cnt"] = foreign_df["amt"] / foreign_df["cnt"]

    return foreign_df.reset_index(drop=True)


def save_processed(foreign_df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    foreign_df.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from data_loader import load_raw_data, validate_business_categories

    raw = load_raw_data()
    foreign_df = clean_and_filter_foreign(raw)
    validate_business_categories(foreign_df)  # 정규화 후 11종 재검증
    save_processed(foreign_df)

    print(f"[OK] 외국인 데이터 {len(foreign_df):,}행 추출 -> {PROCESSED_PATH}")
    print(f"[OK] 시군구 수: {foreign_df['CCG_NM'].nunique()}, 업종 수: {foreign_df['TP_BUZ_NM'].nunique()}")
    print(f"[OK] 총 소비금액: {foreign_df['amt'].sum():,}원 / 총 건수: {foreign_df['cnt'].sum():,}건")
