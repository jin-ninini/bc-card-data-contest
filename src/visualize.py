"""
Step 7: 시각화 모듈

matplotlib 한글 폰트 설정과, 노트북 06에서 사용할 8종 차트 생성 함수를 담는다.
색상은 dataviz 스킬의 검증된 기본 팔레트(카테고리 슬롯 1~3: blue/orange/aqua,
전체 3개 조합에서 CVD-safe 검증 통과)를 그대로 사용하고, 3개 세그먼트에
고정 순서로 매핑한다 (색상이 순위가 아니라 세그먼트 정체성을 따르도록).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SEG_A, SEG_B, SEG_C = "생활밀착형", "로컬미식형", "프리미엄외식형"

# dataviz 스킬 참조 팔레트: 카테고리 슬롯 1(blue)/2(orange)/3(aqua) — 3개 조합 all-pairs 검증 통과
SEGMENT_COLORS = {
    SEG_A: "#2a78d6",   # slot 1: blue
    SEG_B: "#eb6834",   # slot 2: orange
    SEG_C: "#1baf7a",   # slot 3: aqua
}
SEGMENT_ORDER = [SEG_A, SEG_B, SEG_C]

# 차트 크롬(축·그리드·잉크) — 라이트 모드 고정
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# 순차(sequential) 팔레트: 지도·히트맵 등 크기 인코딩에 사용하는 blue 램프
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def setup_korean_font() -> str:
    """시스템에서 사용 가능한 한글 폰트를 탐색해 matplotlib 기본 폰트로 설정한다.

    우선순위: Noto Sans CJK KR > NanumGothic > Malgun Gothic > 시스템 sans-serif
    (플랫폼마다 설치된 폰트가 달라 하드코딩 대신 fontManager에서 실제 탐색한다).
    """
    preferred = ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]
    available = {f.name for f in fm.fontManager.ttflist}

    chosen = next((name for name in preferred if name in available), None)
    if chosen is None:
        # 이름 매칭 실패 시, 파일명에 CJK/Noto가 포함된 폰트를 폭넓게 탐색
        candidates = [f for f in fm.fontManager.ttflist if "CJK" in f.name or "Noto Sans" in f.name]
        chosen = candidates[0].name if candidates else "sans-serif"

    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용 시 마이너스 기호 깨짐 방지
    return chosen


def _style_axis(ax):
    """공통 차트 크롬: 무채색 축, 옅은 그리드, 상단/우측 테두리 제거."""
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, filename: str, dpi: int = 300):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. 세그먼트별 업종 구성비 비교 (누적 막대)
# ---------------------------------------------------------------------------
def plot_segment_industry_composition(segment_df: pd.DataFrame, buz_list: list[str]) -> Path:
    profile = segment_df.groupby("segment")[[f"buz_pct_{b}" for b in buz_list]].mean()
    profile = profile.reindex(SEGMENT_ORDER)
    profile.columns = buz_list

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(profile))
    cmap = plt.get_cmap("tab20")
    for i, buz in enumerate(buz_list):
        vals = profile[buz].values
        ax.bar(profile.index, vals, bottom=bottom, label=buz, color=cmap(i / len(buz_list)),
               edgecolor=SURFACE, linewidth=1.5)
        bottom += vals

    _style_axis(ax)
    ax.set_ylabel("업종별 소비금액 비중 (%)", color=INK_SECONDARY)
    ax.set_title("세그먼트별 업종 구성비 비교 (평균)", color=INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=9)
    return _save(fig, "01_segment_industry_composition.png")


# ---------------------------------------------------------------------------
# 2. 세그먼트별 연령대 분포 비교
# ---------------------------------------------------------------------------
def plot_segment_age_distribution(segment_df: pd.DataFrame, age_list: list[str]) -> Path:
    profile = segment_df.groupby("segment")[[f"age_pct_{a}" for a in age_list]].mean().reindex(SEGMENT_ORDER)
    profile.columns = age_list

    x = np.arange(len(age_list))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, seg in enumerate(SEGMENT_ORDER):
        ax.bar(x + (i - 1) * width, profile.loc[seg].values, width=width, label=seg,
               color=SEGMENT_COLORS[seg])

    _style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(age_list)
    ax.set_ylabel("연령대별 소비금액 비중 (%)", color=INK_SECONDARY)
    ax.set_title("세그먼트별 연령대 분포 비교 (평균)", color=INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    return _save(fig, "02_segment_age_distribution.png")


# ---------------------------------------------------------------------------
# 3. 월별 소비 추이 비교 (1월=100 지수화)
# ---------------------------------------------------------------------------
def plot_monthly_trend(foreign_df: pd.DataFrame, segment_df: pd.DataFrame) -> Path:
    merged = foreign_df.merge(segment_df[["SIDO_NM", "CCG_NM", "segment"]], on=["SIDO_NM", "CCG_NM"])
    monthly = merged.groupby(["segment", "STRD_YYMM"])["amt"].sum().unstack("STRD_YYMM").reindex(SEGMENT_ORDER)
    indexed = monthly.div(monthly.iloc[:, 0], axis=0) * 100  # 1월 = 100 지수화

    fig, ax = plt.subplots(figsize=(9, 6))
    for seg in SEGMENT_ORDER:
        ax.plot(indexed.columns, indexed.loc[seg], marker="o", markersize=6, linewidth=2,
                color=SEGMENT_COLORS[seg], label=seg)

    _style_axis(ax)
    ax.axhline(100, color=INK_MUTED, linewidth=1, linestyle="--", zorder=0)
    ax.set_ylabel("월별 소비금액 지수 (1월=100)", color=INK_SECONDARY)
    ax.set_title("세그먼트별 월별 소비 추이 (1월=100 지수화)", color=INK_PRIMARY, fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    return _save(fig, "03_monthly_trend.png")


# ---------------------------------------------------------------------------
# 4. 군집분석 결과 산점도 (PCA 2차원, 실루엣 스코어 병기)
# ---------------------------------------------------------------------------
def plot_cluster_scatter(segment_df: pd.DataFrame, silhouette: float, best_k: int) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    for seg in SEGMENT_ORDER:
        sub = segment_df[segment_df["segment"] == seg]
        ax.scatter(sub["pca_1"], sub["pca_2"], s=45, alpha=0.75, color=SEGMENT_COLORS[seg],
                   label=f"{seg} (n={len(sub)})", edgecolor="white", linewidth=0.5)

    _style_axis(ax)
    ax.set_xlabel("PCA 1", color=INK_SECONDARY)
    ax.set_ylabel("PCA 2", color=INK_SECONDARY)
    ax.set_title(
        f"군집분석 결과 (K={best_k}, 실루엣 스코어={silhouette:.3f})",
        color=INK_PRIMARY, fontsize=13, fontweight="bold",
    )
    ax.legend(frameon=False)
    return _save(fig, "04_cluster_scatter_pca.png")


# ---------------------------------------------------------------------------
# 5. 세그먼트별 체류자격 구성비 비교 ⭐ 핵심 시각화 (Step 4 결과)
# ---------------------------------------------------------------------------
def plot_visa_composition_by_segment(merged_df: pd.DataFrame, visa_groups: list[str]) -> Path:
    profile = merged_df.groupby("segment")[[f"{g}_비중" for g in visa_groups]].mean().reindex(SEGMENT_ORDER)
    profile.columns = visa_groups

    x = np.arange(len(visa_groups))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, seg in enumerate(SEGMENT_ORDER):
        ax.bar(x + (i - 1) * width, profile.loc[seg].values, width=width, label=seg,
               color=SEGMENT_COLORS[seg])

    _style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(visa_groups)
    ax.set_ylabel("체류자격 그룹 비중 (%, 등록외국인 수 기준)", color=INK_SECONDARY)
    ax.set_title(
        "세그먼트별 체류자격 구성비 비교 (법무부 등록외국인 현황 교차분석)",
        color=INK_PRIMARY, fontsize=13, fontweight="bold",
    )
    ax.legend(frameon=False)
    return _save(fig, "05_visa_composition_by_segment.png")


# ---------------------------------------------------------------------------
# 6. 상위 소비 지역 Top 15 (세그먼트별 색상 구분)
# ---------------------------------------------------------------------------
def plot_top15_regions(segment_df: pd.DataFrame) -> Path:
    top15 = segment_df.nlargest(15, "총소비금액").copy()
    top15["지역명"] = top15["SIDO_NM"] + " " + top15["CCG_NM"]
    top15 = top15.sort_values("총소비금액")

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [SEGMENT_COLORS[s] for s in top15["segment"]]
    ax.barh(top15["지역명"], top15["총소비금액"] / 1e8, color=colors)

    _style_axis(ax)
    ax.set_xlabel("총 소비금액 (억원)", color=INK_SECONDARY)
    ax.set_title("외국인 소비금액 상위 15개 시군구", color=INK_PRIMARY, fontsize=13, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS[s]) for s in SEGMENT_ORDER]
    ax.legend(handles, SEGMENT_ORDER, frameon=False, loc="lower right")
    return _save(fig, "06_top15_regions.png")


# ---------------------------------------------------------------------------
# 7. 전국 시군구 세그먼트 지도 (folium)
# ---------------------------------------------------------------------------
# 시군구 실좌표: 소상공인시장진흥공단 상가(상권)정보 API(B553077/api/open/sdsc2, 공공데이터포털
# 15012005)로 시군구별 음식업종 상가업소 표본을 수집해 만든 실제 중심좌표 (src/store_data.py).
# 255개 목표 지역 중 252개(98.8%) 확보. 나머지 3개(인천 중구·동구·서구)는 이 API의 기준시점
# (2026.6)에 이미 제물포구·영종구·서해구 등으로 행정구역이 개편되어 옛 이름으로는 데이터가
# 없다 — 실좌표가 없는 지역만 시도 대표좌표 + 지터로 대체(fallback)한다.
STORE_CENTROID_PATH = PROJECT_ROOT / "data" / "external" / "store_centroids.csv"

# fallback용 시도 대표 좌표 (실좌표 미확보 지역에만 사용)
SIDO_CENTROIDS = {
    "서울특별시": (37.5665, 126.9780), "부산광역시": (35.1796, 129.0756),
    "대구광역시": (35.8714, 128.6014), "인천광역시": (37.4563, 126.7052),
    "광주광역시": (35.1595, 126.8526), "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114), "세종특별자치시": (36.4801, 127.2891),
    "경기도": (37.4138, 127.5183), "강원특별자치도": (37.8228, 128.1555),
    "충청북도": (36.6357, 127.4917), "충청남도": (36.5184, 126.8000),
    "전라북도": (35.7175, 127.1530), "전북특별자치도": (35.7175, 127.1530),
    "전라남도": (34.8679, 126.9910), "경상북도": (36.4919, 128.8889),
    "경상남도": (35.4606, 128.2132), "제주특별자치도": (33.4890, 126.4983),
}


def _region_coords(segment_df: pd.DataFrame) -> pd.DataFrame:
    """세그먼트 테이블에 지도용 좌표(lat, lon, is_real_coord)를 붙인다.

    store_centroids.csv(실좌표 캐시)가 있으면 우선 사용하고, 캐시에 없는 지역만
    시도 대표좌표 + 지터(재현성 고정 시드)로 대체한다.
    """
    df = segment_df.copy()
    rng = np.random.RandomState(42)

    if STORE_CENTROID_PATH.exists():
        centroids = pd.read_csv(STORE_CENTROID_PATH)[["SIDO_NM", "CCG_NM", "lon", "lat"]]
        df = df.merge(centroids, on=["SIDO_NM", "CCG_NM"], how="left")
    else:
        df["lon"] = np.nan
        df["lat"] = np.nan

    df["is_real_coord"] = df["lat"].notna()

    missing = df["lat"].isna()
    for idx in df.index[missing]:
        base = SIDO_CENTROIDS.get(df.loc[idx, "SIDO_NM"])
        if base is None:
            continue
        df.loc[idx, "lat"] = base[0] + rng.uniform(-0.25, 0.25)
        df.loc[idx, "lon"] = base[1] + rng.uniform(-0.25, 0.25)

    return df


def plot_segment_map(segment_df: pd.DataFrame) -> Path:
    import folium

    df = _region_coords(segment_df)
    n_real = df["is_real_coord"].sum()
    m = folium.Map(location=[36.2, 127.9], zoom_start=7, tiles="OpenStreetMap")

    max_amt = df["총소비금액"].max()
    for _, row in df.iterrows():
        if pd.isna(row["lat"]):
            continue
        radius = 3 + 15 * (row["총소비금액"] / max_amt) ** 0.5
        coord_note = "" if row["is_real_coord"] else " (근사좌표)"
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=radius,
            color=SEGMENT_COLORS[row["segment"]],
            fill=True,
            fill_color=SEGMENT_COLORS[row["segment"]],
            fill_opacity=0.75,
            weight=1,
            popup=f"{row['SIDO_NM']} {row['CCG_NM']}{coord_note} | {row['segment']} | "
                  f"{row['총소비금액']/1e8:.1f}억원",
        ).add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: {SURFACE}; padding: 10px 14px; border-radius: 6px;
                border: 1px solid {GRIDLINE}; font-size: 13px; color: {INK_PRIMARY};">
      <b>세그먼트</b><br>
      {"".join(f'<span style="color:{SEGMENT_COLORS[s]};">●</span> {s}<br>' for s in SEGMENT_ORDER)}
      <hr style="margin:6px 0;">
      <span style="font-size:11px; color:{INK_MUTED};">실좌표 {n_real}/{len(df)}개 지역
      (소상공인시장진흥공단 상가정보 API)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "07_segment_map.html"
    m.save(str(path))
    return path


def plot_segment_map_static(segment_df: pd.DataFrame) -> Path:
    """folium 지도의 정적(PNG) 버전.

    최종 제출물은 PDF 변환이 필요한데(§1) 인터랙티브 HTML은 PDF에 담을 수 없으므로,
    동일한 좌표(실좌표 우선, 미확보 지역만 시도 대표좌표+지터)를 matplotlib 산점도로도 제공한다.
    """
    df = _region_coords(segment_df)
    n_real = df["is_real_coord"].sum()
    fig, ax = plt.subplots(figsize=(7, 9))
    max_amt = df["총소비금액"].max()

    for _, row in df.iterrows():
        if pd.isna(row["lat"]):
            continue
        size = 15 + 300 * (row["총소비금액"] / max_amt)
        marker = "o" if row["is_real_coord"] else "x"  # 근사좌표는 x로 구분 표시
        ax.scatter(row["lon"], row["lat"], s=size, color=SEGMENT_COLORS[row["segment"]], alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3, marker=marker)

    ax.set_facecolor(SURFACE)
    ax.set_xlim(124.5, 130.0)
    ax.set_ylim(33.0, 39.0)
    ax.set_aspect(1.4)
    ax.axis("off")
    ax.set_title(
        "전국 시군구 세그먼트 지도 (원 크기 = 총소비금액)",
        color=INK_PRIMARY, fontsize=13, fontweight="bold",
    )
    ax.text(
        0.5, -0.02, f"좌표 출처: 소상공인시장진흥공단 상가(상권)정보 API (실좌표 {n_real}/{len(df)}개 지역, x표시=근사좌표)",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color=INK_MUTED,
    )
    handles = [plt.scatter([], [], s=80, color=SEGMENT_COLORS[s], label=s) for s in SEGMENT_ORDER]
    ax.legend(handles=handles, loc="lower left", frameon=False)
    return _save(fig, "07_segment_map.png")


def generate_all_figures(foreign_df, segment_df, merged_visa_df, buz_list, age_list, visa_groups, silhouette, best_k):
    """노트북 06에서 호출하는 일괄 생성 진입점."""
    setup_korean_font()
    paths = [
        plot_segment_industry_composition(segment_df, buz_list),
        plot_segment_age_distribution(segment_df, age_list),
        plot_monthly_trend(foreign_df, segment_df),
        plot_cluster_scatter(segment_df, silhouette, best_k),
        plot_visa_composition_by_segment(merged_visa_df, visa_groups),
        plot_top15_regions(segment_df),
        plot_segment_map_static(segment_df),
        plot_segment_map(segment_df),
    ]
    return paths
