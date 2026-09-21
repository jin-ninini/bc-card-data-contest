"""
Step 3: 3대 세그먼트 분류

(A) 규칙기반 스코어링과 (B) 군집분석(K-means) 두 가지 방법으로 각 시군구를
분류하고, (C) 두 결과를 교차검증한 뒤 최종 세그먼트 라벨을 부여한다.

공식 세그먼트 라벨 (요구사항 §3-2, 이 세 라벨만 산출물에 사용한다):
  - 생활밀착형 (거주 중심)
  - 로컬미식형 (관광 중심)
  - 프리미엄외식형 (관광 중심)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from feature_engineering import AGE_LABEL_LIST, BUZ_LIST, REGION_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEG_A = "생활밀착형"
SEG_B = "로컬미식형"
SEG_C = "프리미엄외식형"

# --- 규칙기반 스코어링에 쓰이는 업종 그룹 정의 ---
# 근거: 11개 업종 중 소매/편의 3종(슈퍼마켓·편의점·대형할인점)과
# 정통 외식 업종을 대비시켜 "생필품 반복구매형" 성향을 수치화한다.
# "외식 5종"은 6개 식사류 업종(서양음식·일반한식·일식회집·한정식·갈비전문점·중국음식) 중
# 중국음식을 제외한 5개다. 중국음식은 한식지향/양식지향 어디에도 속하지 않는
# 독립적 성격(§3-3의 영등포구 사례처럼 국지적 편중이 큰 업종)이라
# 생활밀착지수 계산에서도 별도로 취급(제외)한다. 제과점·스넥 역시
# "생필품"도 "정통 외식"도 아닌 중간적 성격이라 두 지수 계산에서 제외했다.
RETAIL_3 = ["슈퍼마켓", "편의점", "대형할인점"]
DINING_5 = ["서양음식", "일반한식", "일식회집", "한정식", "갈비전문점"]
HANSIK_BUZ = ["일반한식", "한정식", "갈비전문점"]
YANGSIK_BUZ = ["서양음식", "일식회집"]

RANDOM_STATE = 42


def compute_rule_scores(features: pd.DataFrame) -> pd.DataFrame:
    """(A) 규칙기반 스코어링: 생활밀착지수/한식지향지수/양식지향지수 계산 후 1차 라벨링."""
    df = features.copy()
    df["생활밀착지수"] = df[[f"buz_pct_{b}" for b in RETAIL_3]].sum(axis=1) - df[
        [f"buz_pct_{b}" for b in DINING_5]
    ].sum(axis=1)
    df["한식지향지수"] = df[[f"buz_pct_{b}" for b in HANSIK_BUZ]].sum(axis=1)
    df["양식지향지수"] = df[[f"buz_pct_{b}" for b in YANGSIK_BUZ]].sum(axis=1)

    # 1차 라벨링 규칙 (분포 기반):
    #  - 생활밀착지수 > 0  → 소매/편의 비중이 정통 외식 비중을 초과하는 지역 → 생활밀착형
    #    (0을 기준으로 삼는 이유: 두 지수가 같은 척도(비중 %)의 차이이므로
    #     0 초과 여부가 "소매 우위 vs 외식 우위"를 가르는 구조적으로 의미 있는 경계다.
    #     임의의 분위수보다 해석이 명확해 채택했다.)
    #  - 나머지(외식 우위) 지역은 한식지향지수 vs 양식지향지수 비교로 B/C 분리
    df["rule_segment"] = np.where(
        df["생활밀착지수"] > 0,
        SEG_A,
        np.where(df["한식지향지수"] >= df["양식지향지수"], SEG_B, SEG_C),
    )
    return df


def run_kmeans_clustering(features: pd.DataFrame, k_range=range(2, 9)) -> tuple[pd.DataFrame, dict]:
    """(B) 군집분석: 업종 11차원 + 연령 6차원 표준화 후 K-means.

    K는 실루엣 스코어가 최대인 값을 채택한다 (K=3 강제하지 않음).
    """
    feature_cols = [f"buz_pct_{b}" for b in BUZ_LIST] + [f"age_pct_{a}" for a in AGE_LABEL_LIST]
    X = StandardScaler().fit_transform(features[feature_cols])

    silhouette_by_k = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        silhouette_by_k[k] = silhouette_score(X, km.labels_)

    best_k = max(silhouette_by_k, key=silhouette_by_k.get)

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X)
    df = features.copy()
    df["cluster"] = km_final.labels_
    df["pca_1"], df["pca_2"] = _pca_2d(X)

    meta = {
        "silhouette_by_k": silhouette_by_k,
        "best_k": best_k,
        "best_silhouette": silhouette_by_k[best_k],
        "feature_cols": feature_cols,
        "X_scaled": X,
        "kmeans_model": km_final,
    }
    return df, meta


def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    return coords[:, 0], coords[:, 1]


def map_clusters_to_segments(df_with_cluster: pd.DataFrame) -> dict:
    """군집 번호(0,1,2,...) -> 공식 세그먼트 라벨 매핑을 각 군집의 평균 지표로 결정한다.

    ⚠️ 군집 내부에서 "한식지향지수 vs 양식지향지수" 절대값을 비교하면 안 된다.
    실측 결과, 프리미엄외식형에 해당하는 군집조차 한식지향지수 절대값이 양식지향지수보다
    높게 나온다 (외식 자체가 드물어 두 지수 모두 낮은 가운데, 그나마 한식 비중이 남아있기
    때문). 세 군집을 "서로 비교"했을 때 상대적으로 어느 지수가 튀어 오르는지를 봐야
    §3-2가 말하는 "일반한식 비중 압도적"/"서양음식 비중 압도적" 성격이 드러난다.

    규칙:
      1) 생활밀착지수 평균이 가장 높은 군집 -> 생활밀착형
      2) 나머지 군집 중 한식지향지수 평균이 가장 높은 군집 -> 로컬미식형
      3) 그 나머지(K=3이면 1개, K>3이면 여러 개) -> 프리미엄외식형
    군집 수가 3이 아닌 경우 이 규칙은 "생활밀착형 1개 + 로컬미식형 1개 + 나머지 전부
    프리미엄외식형"으로 단순화되며, 이는 §9-3에서 말하는 확인 필요 상황이므로
    노트북에서 K!=3일 때 명시적 경고를 출력한다.
    """
    profile = df_with_cluster.groupby("cluster")[["생활밀착지수", "한식지향지수", "양식지향지수"]].mean()
    mapping = {}
    life_cluster = profile["생활밀착지수"].idxmax()
    mapping[life_cluster] = SEG_A

    remaining = profile.drop(index=life_cluster)
    local_food_cluster = remaining["한식지향지수"].idxmax()
    mapping[local_food_cluster] = SEG_B

    for c in remaining.index:
        if c == local_food_cluster:
            continue
        mapping[c] = SEG_C
    return mapping


def classify_segments(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """전체 파이프라인: 규칙기반 + 군집분석 + 상호검증 + 최종 라벨 부여."""
    rule_df = compute_rule_scores(features)
    cluster_df, meta = run_kmeans_clustering(rule_df)

    cluster_to_seg = map_clusters_to_segments(cluster_df)
    cluster_df["segment"] = cluster_df["cluster"].map(cluster_to_seg)

    crosstab = pd.crosstab(cluster_df["rule_segment"], cluster_df["segment"])
    agreement_rate = (cluster_df["rule_segment"] == cluster_df["segment"]).mean()

    meta["cluster_to_segment"] = cluster_to_seg
    meta["crosstab"] = crosstab
    meta["agreement_rate"] = agreement_rate

    return cluster_df, meta


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from feature_engineering import build_region_features

    raw = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "foreign_consumption_clean.csv", dtype={"STRD_YYMM": str})
    features = build_region_features(raw)
    result, meta = classify_segments(features)

    print(f"[OK] 실루엣 스코어 K별: {meta['silhouette_by_k']}")
    print(f"[OK] 최적 K: {meta['best_k']} (실루엣={meta['best_silhouette']:.4f})")
    print(f"[OK] 군집->세그먼트 매핑: {meta['cluster_to_segment']}")
    print(f"[OK] 규칙기반 vs 군집 일치도: {meta['agreement_rate']*100:.1f}%")
    print(meta["crosstab"])
    print(result["segment"].value_counts())
