"""
Step 4: 법무부 「등록외국인 지역별 현황」 외부 공공데이터 로딩 및 재범주화

출처: 법무부 출입국·외국인정책본부 통계월보 2026년 6월호
      게시물 "등록외국인 지역별 현황(2026년 6월말 기준)"
      (moj.go.kr/bbs/immigration/227/608715, 첨부 "2.3. 등록외국인(지역, 자격) 현황(6월말).xlsx")
      공공누리 4유형 (출처표시 + 상업적 이용금지 + 변경금지)
기준시점: 2026-06-30 (카드데이터의 마지막 집계월 202606과 동일 시점 — 시차 없음)

⚠️ 자료 구조상 한계: 이 표에는 F-4(재외동포) 자격이 없다. 재외동포는 일반적인
"외국인등록"이 아니라 별도의 "국내거소신고" 제도로 관리되기 때문에 본 표(등록외국인
현황)에는 애초에 집계되지 않는다 (별도 게시물 "외국국적동포 거소신고 현황"으로 관리됨).
따라서 정주계열 그룹에는 F-5(영주)·F-6(결혼이민)만 포함하고, F-4는 한계점으로 보고서에 명시한다.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from region_mapping import map_card_region_to_moj
MOJ_XLSX_PATH = PROJECT_ROOT / "data" / "external" / "등록외국인_지역_자격_대분류_202606.xlsx"

# 체류자격 대분류 컬럼 -> 분석용 그룹 재범주화 (요구사항 §4 예시 기반)
VISA_GROUPS: dict[str, list[str]] = {
    # 취업계열: 비전문/숙련 취업 비자 (E-7, E-9 중심 + 관련 취업 비자 전반)
    "취업계열": [
        "E7(특정활동)", "E8(계절근로)", "E9(비전문취업)", "E10(선원취업)", "H2(방문취업)",
    ],
    # 유학계열: 유학·일반연수
    "유학계열": ["D2(유학)", "D4(일반연수)"],
    # 정주계열: 영주·결혼이민 (F-4는 자료 구조상 미포함 — 위 docstring 참조)
    "정주계열": ["F5(영주)", "F6(결혼이민)"],
}
ALL_GROUP_COLS = [c for cols in VISA_GROUPS.values() for c in cols]

# 카드데이터 feature_engineering.LOW_SAMPLE_QUANTILE(0.10)과 동일 기준을
# 법무부 등록외국인 총합계에도 적용해 저표본 지역을 동일한 방식으로 정의한다.
LOW_MOJ_SAMPLE_QUANTILE = 0.10


def load_moj_visa_by_region(path: Path = MOJ_XLSX_PATH) -> pd.DataFrame:
    """법무부 시군구x체류자격(대분류) xlsx를 읽어 지역별 성별-총계 행만 추출한다."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header = list(raw.iloc[2, :])
    df = raw.iloc[3:].copy()
    df.columns = header
    df = df.reset_index(drop=True)

    # '총합계' 전국 행, 시도별 소계('총계') 행 제외 -> 실제 시군구 단위 행만 남김.
    # 단, 세종특별자치시는 시군구 세분류가 없어 시군구 컬럼이 빈 문자열('')로 표기되는데
    # 이는 region_mapping.EXPLICIT_OVERRIDES가 기대하는 형태이므로 제외하지 않는다.
    df = df[df["시도"] != "총합계"]
    df = df[df["성별"] == "총계"].copy()
    # 엑셀의 빈 셀(세종특별자치시 시군구 칸)이 공백 문자 ' '로 읽히는 경우가 있어 strip 정규화
    df["시도"] = df["시도"].str.strip()
    df["시군구"] = df["시군구"].str.strip()
    df = df[df["시군구"] != "총계"]

    missing_cols = set(ALL_GROUP_COLS) - set(df.columns)
    assert not missing_cols, f"법무부 자료에 기대한 체류자격 컬럼이 없습니다: {missing_cols}"

    numeric_cols = header[3:]
    df[numeric_cols] = df[numeric_cols].astype(float).fillna(0.0)
    return df.reset_index(drop=True)


def compute_visa_group_shares(moj_df: pd.DataFrame) -> pd.DataFrame:
    """지역별 체류자격 그룹(취업/유학/정주) 인원수 및 비중(%)을 계산한다."""
    df = moj_df[["시도", "시군구", "총합계"]].copy()
    for group_name, cols in VISA_GROUPS.items():
        df[f"{group_name}_인원"] = moj_df[cols].sum(axis=1)
    for group_name in VISA_GROUPS:
        df[f"{group_name}_비중"] = df[f"{group_name}_인원"] / df["총합계"] * 100

    # 법무부 총합계(등록외국인 수)는 지역별 편차가 극단적으로 크다(실측 최소 1명 ~
    # 최대 40,479명). 총합계가 매우 작은 지역은 비중(%) 값이 개인 1~2명 단위로도
    # 크게 흔들려(예: 총합계=1인 지역은 그 1명의 자격이 곧바로 100%/0%가 됨) 통계
    # 검정에 노이즈를 더할 수 있다. 카드데이터 쪽에 이미 있는 `low_sample`(하위
    # 10분위) 플래그와 동일한 기준을 법무부 쪽에도 적용해, 강건성 검증(재검정) 시
    # 이 지역들을 제외해볼 수 있도록 플래그만 붙여 둔다(기본 분석에서 제외하지는
    # 않음 — 배제 여부는 external_data.run_visa_validation_robustness에서 결정).
    threshold = df["총합계"].quantile(LOW_MOJ_SAMPLE_QUANTILE)
    df["moj_low_sample"] = df["총합계"] < threshold
    df.attrs["moj_low_sample_threshold"] = threshold
    return df


def merge_segment_with_visa(segment_df: pd.DataFrame, visa_share_df: pd.DataFrame) -> pd.DataFrame:
    """세그먼트 분류 결과(카드데이터 기준)에 법무부 체류자격 비중(지역 매핑 적용)을 결합한다."""
    seg = segment_df.copy()
    mapped = seg.apply(lambda r: map_card_region_to_moj(r["SIDO_NM"], r["CCG_NM"]), axis=1)
    seg["moj_시도"] = [m[0] for m in mapped]
    seg["moj_시군구"] = [m[1] for m in mapped]

    merged = seg.merge(
        visa_share_df,
        left_on=["moj_시도", "moj_시군구"],
        right_on=["시도", "시군구"],
        how="left",
        indicator=True,
    )
    n_fail = (merged["_merge"] != "both").sum()
    assert n_fail == 0, f"세그먼트-법무부 지역 조인 실패 {n_fail}건 발생 (region_mapping.py 점검 필요)"
    return merged.drop(columns=["_merge"])


def run_visa_validation(merged_df: pd.DataFrame, seg_a_label: str = "생활밀착형") -> pd.DataFrame:
    """핵심 검증 질문: 생활밀착형 지역이 다른 세그먼트 대비 취업계열 체류자격 비중이
    통계적으로 유의하게 높은가?

    표본 크기가 세그먼트마다 다르고(45~129개) 비중(%) 데이터가 정규분포를 보장하지
    않으므로 비모수 검정인 Mann-Whitney U 검정을 사용한다 (2집단 비교, 등분산 가정 불필요).
    """
    from scipy import stats

    rows = []
    for group_name in VISA_GROUPS:
        col = f"{group_name}_비중"
        a_vals = merged_df.loc[merged_df["segment"] == seg_a_label, col].dropna()
        other_vals = merged_df.loc[merged_df["segment"] != seg_a_label, col].dropna()

        u_stat, p_value = stats.mannwhitneyu(a_vals, other_vals, alternative="two-sided")

        rows.append({
            "체류자격그룹": group_name,
            "비교": f"{seg_a_label} vs 나머지",
            f"{seg_a_label}_평균": a_vals.mean(),
            f"{seg_a_label}_중앙값": a_vals.median(),
            "나머지_평균": other_vals.mean(),
            "나머지_중앙값": other_vals.median(),
            "검정방법": "Mann-Whitney U",
            "통계량": u_stat,
            "p_value": p_value,
            "유의(p<0.05)": p_value < 0.05,
        })

    # 참고용: 3개 세그먼트 전체 비교 (Kruskal-Wallis)
    for group_name in VISA_GROUPS:
        col = f"{group_name}_비중"
        samples = [g[col].dropna().values for _, g in merged_df.groupby("segment")]
        h_stat, p_value = stats.kruskal(*samples)
        rows.append({
            "체류자격그룹": group_name,
            "비교": "3개 세그먼트 전체",
            f"{seg_a_label}_평균": None,
            f"{seg_a_label}_중앙값": None,
            "나머지_평균": None,
            "나머지_중앙값": None,
            "검정방법": "Kruskal-Wallis",
            "통계량": h_stat,
            "p_value": p_value,
            "유의(p<0.05)": p_value < 0.05,
        })

    return pd.DataFrame(rows)


def run_visa_validation_robustness(
    merged_df: pd.DataFrame, seg_a_label: str = "생활밀착형"
) -> pd.DataFrame:
    """강건성 점검: 법무부 저표본(moj_low_sample) 지역을 제외해도 핵심 결론이 유지되는지 재검정한다.

    `merge_segment_with_visa` 결과에는 `moj_low_sample` 플래그가 이미 포함되어 있다
    (compute_visa_group_shares에서 생성). 이 지역들(등록외국인 총합계 하위 10분위,
    예: 1명인 지역도 존재)을 제외한 뒤 동일한 Mann-Whitney U 검정을 다시 수행해,
    원본 결과가 소수의 불안정한 비중값 때문에 만들어진 착시가 아님을 확인한다.
    """
    from scipy import stats

    assert "moj_low_sample" in merged_df.columns, (
        "merged_df에 moj_low_sample 컬럼이 없습니다. "
        "compute_visa_group_shares() 결과를 merge_segment_with_visa()에 넣었는지 확인하세요."
    )
    robust_df = merged_df.loc[~merged_df["moj_low_sample"]]
    n_excluded = int(merged_df["moj_low_sample"].sum())

    rows = []
    for group_name in VISA_GROUPS:
        col = f"{group_name}_비중"
        a_vals = robust_df.loc[robust_df["segment"] == seg_a_label, col].dropna()
        other_vals = robust_df.loc[robust_df["segment"] != seg_a_label, col].dropna()
        u_stat, p_value = stats.mannwhitneyu(a_vals, other_vals, alternative="two-sided")
        rows.append({
            "체류자격그룹": group_name,
            "제외지역수": n_excluded,
            "잔여지역수": len(robust_df),
            f"{seg_a_label}_평균(저표본제외)": a_vals.mean(),
            "나머지_평균(저표본제외)": other_vals.mean(),
            "p_value(저표본제외)": p_value,
            "유의(p<0.05)": p_value < 0.05,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    moj = load_moj_visa_by_region()
    print(f"[OK] 법무부 자료 지역 수(시군구, 총계 행 기준): {len(moj)}")
    shares = compute_visa_group_shares(moj)
    print(shares[["시도", "시군구", "총합계", "취업계열_비중", "유학계열_비중", "정주계열_비중"]].head(10))
    print(shares[[f"{g}_비중" for g in VISA_GROUPS]].describe())
