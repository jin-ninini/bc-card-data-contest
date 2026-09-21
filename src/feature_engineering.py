"""
Step 2: 시군구별 특징 벡터 생성

외국인 소비 clean 데이터를 (SIDO_NM, CCG_NM) 단위로 집계하여
세그먼트 분류(Step 3)에 사용할 피처 테이블을 만든다.

⚠️ 중요: CCG_NM 만으로 groupby 하면 안 된다. "중구"·"동구"·"서구"·"남구"·"북구"·
"강서구"·"고성군" 등 7개 시군구명은 서로 다른 광역시도에 동시에 존재하므로
반드시 (SIDO_NM, CCG_NM) 복합키로 묶어야 한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGION_KEYS = ["SIDO_NM", "CCG_NM"]

BUZ_LIST = [
    "서양음식", "일반한식", "중국음식", "일식회집", "한정식",
    "갈비전문점", "편의점", "대형할인점", "슈퍼마켓", "제과점", "스넥",
]
AGE_LABEL_LIST = ["20대 이하", "20대", "30대", "40대", "50대", "60대 이상"]

# 저표본(low_sample) 판정 임계치.
# 근거: 255개 시군구의 6개월 합산 이용건수 분포에서 하위 10분위수를 사용.
# 실측 데이터에서는 최솟값(4,315건)이 0에 가깝게 낮지 않아 "결측"이 아니라
# "표본이 적어 비중·군집 결과가 불안정할 수 있는" 지역을 걸러내는 용도이므로,
# 스펙이 제시한 하위 10~25분위 범위 중 보수적인(적게 걸러내는) 10분위를 채택한다.
LOW_SAMPLE_QUANTILE = 0.10


def _pct_pivot(df: pd.DataFrame, category_col: str, categories: list[str], prefix: str) -> pd.DataFrame:
    """지역별 category_col 값별 amt 비중(%) 피벗 테이블을 만든다."""
    pivot = (
        df.pivot_table(index=REGION_KEYS, columns=category_col, values="amt", aggfunc="sum", fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct.columns = [f"{prefix}{c}" for c in pct.columns]
    return pct


def _monthly_cv(df: pd.DataFrame) -> pd.Series:
    """지역별 월간 소비금액의 변동계수(CV = std/mean)를 계산한다.

    CV가 높을수록 특정 월에 소비가 몰리는(계절성·관광 성수기형) 지역,
    낮을수록 월별로 고르게 소비가 발생하는(거주형) 지역으로 해석한다.
    표본이 6개월(=6개 관측치)로 적어 표본표준편차(ddof=1)를 사용한다.
    """
    monthly = df.pivot_table(index=REGION_KEYS, columns="STRD_YYMM", values="amt", aggfunc="sum", fill_value=0)
    return monthly.std(axis=1, ddof=1) / monthly.mean(axis=1)


def build_region_features(foreign_df: pd.DataFrame) -> pd.DataFrame:
    """(SIDO_NM, CCG_NM) 단위 특징 벡터 테이블을 생성한다."""
    totals = foreign_df.groupby(REGION_KEYS).agg(총소비금액=("amt", "sum"), 총건수=("cnt", "sum")).reset_index()
    totals["건당평균결제금액"] = totals["총소비금액"] / totals["총건수"]

    buz_pct = _pct_pivot(foreign_df, "TP_BUZ_NM", BUZ_LIST, "buz_pct_").reset_index()
    age_pct = _pct_pivot(foreign_df, "AGE_LABEL", AGE_LABEL_LIST, "age_pct_").reset_index()
    cv = _monthly_cv(foreign_df).reset_index(name="월별변동계수CV")

    features = totals.merge(buz_pct, on=REGION_KEYS).merge(age_pct, on=REGION_KEYS).merge(cv, on=REGION_KEYS)

    threshold = features["총건수"].quantile(LOW_SAMPLE_QUANTILE)
    features["low_sample"] = features["총건수"] < threshold
    features.attrs["low_sample_threshold"] = threshold

    return features


if __name__ == "__main__":
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "foreign_consumption_clean.csv", dtype={"STRD_YYMM": str})
    features = build_region_features(df)
    print(f"[OK] 지역 피처 테이블: {features.shape[0]}개 지역 x {features.shape[1]}개 컬럼")
    print(f"[OK] low_sample 임계치(하위 10분위 총건수): {features.attrs['low_sample_threshold']:.0f}건")
    print(f"[OK] low_sample=True 지역 수: {features['low_sample'].sum()}")
    print(features[REGION_KEYS + ["총소비금액", "총건수", "건당평균결제금액", "월별변동계수CV", "low_sample"]].head())
