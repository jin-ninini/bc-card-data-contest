"""
Step 7 지도 시각화 폴리싱: 소상공인시장진흥공단 상가(상권)정보 API 연동

기존 `visualize.plot_segment_map*`은 시군구 실좌표가 없어 "시도 대표좌표 + 무작위 지터"로
근사했다. 이 모듈은 공공데이터포털 소상공인시장진흥공단 상가업소정보 API
(B553077/api/open/sdsc2, https://www.data.go.kr/data/15012005/openapi.do)에서
시군구별 실제 상가업소 위경도를 가져와 각 시군구의 "실제 상권 중심좌표"를 계산한다.
음식(indsLclsCd='I2') 대분류로 조회하면 전국 어디서나 표본이 고르게 분포하고, 카드데이터의
외식 업종들과도 성격이 가장 가깝기 때문에 이 대분류를 좌표 추정용 프록시로 사용한다.

⚠️ API 서비스키는 개인 인증정보이므로 이 파일에 하드코딩하지 않는다.
   환경변수 DATA_GO_KR_SERVICE_KEY로 전달한다.
   (사용법: DATA_GO_KR_SERVICE_KEY=발급받은키 python src/store_data.py)

결과 캐시: data/external/store_centroids.csv (좌표만 저장, 개인정보·인증키 없음 — git 커밋 가능)
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CENTROID_CACHE_PATH = PROJECT_ROOT / "data" / "external" / "store_centroids.csv"

API_BASE = "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInDong"
FOOD_INDS_LCLS_CD = "I2"  # 상권업종대분류 '음식' — 전국에 고르게 분포하는 지리적 프록시

# 카드데이터 SIDO_NM에 대응하는 시도코드(법정동코드 앞 2자리, 특별자치도 개편 반영값을 실측 확인함)
#
# ⚠️ 실측 중 발견한 중요한 사실: 이 상가정보 API(기준월 202606)에는 카드데이터 수집 시점
# 이후의 행정구역 개편이 이미 반영되어 있다.
#   - 광주광역시 + 전라남도 -> "전남광주통합특별시"(ctprvnCd=12)로 통합
#   - 인천광역시 중구·동구·서구 일부 -> 제물포구·영종구·서해구·검단구로 개편
# 카드데이터(SIDO_NM)는 옛 명칭을 그대로 쓰므로, 광주/전남은 통합 코드(12)로 조회하되
# 지역 키는 API가 반환하는 새 시도명이 아니라 "카드데이터 기준 SIDO_NM"으로 강제 매핑한다
# (fetch_region_centroids에서 it["ctprvnNm"] 대신 순회 중인 sido 변수를 사용).
# 인천 중구·동구·서구는 상가정보 API에 더 이상 존재하지 않아 실좌표를 구할 수 없다 —
# 해당 3개 지역은 fallback(시도 대표좌표 지터)으로 남겨두고 한계점으로 명시한다.
SIDO_CTPRVN_CODES = {
    "서울특별시": "11", "부산광역시": "26", "대구광역시": "27", "인천광역시": "28",
    "광주광역시": "12", "대전광역시": "30", "울산광역시": "31", "세종특별자치시": "36",
    "경기도": "41", "강원특별자치도": "51", "충청북도": "43", "충청남도": "44",
    "전북특별자치도": "52", "전라남도": "12", "경상북도": "47", "경상남도": "48",
    "제주특별자치도": "50",
}


def _fetch_page(service_key: str, ctprvn_cd: str, page: int, num_rows: int = 1000, retries: int = 3) -> dict:
    params = {
        "serviceKey": service_key, "divId": "ctprvnCd", "key": ctprvn_cd,
        "indsLclsCd": FOOD_INDS_LCLS_CD, "type": "json",
        "numOfRows": str(num_rows), "pageNo": str(page),
    }
    url = API_BASE + "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.load(resp)
            # 첫 페이지에서 NODATA_ERROR가 뜨는 경우 실제 데이터 없음이 아니라
            # 연속 호출로 인한 일시적 오류인 경우가 실측상 흔해 재시도한다.
            if data["header"]["resultMsg"] == "NODATA_ERROR" and page == 1 and attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            return data
        except urllib.error.URLError as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise last_err if last_err else RuntimeError("알 수 없는 오류")


def fetch_region_centroids(
    service_key: str,
    target_regions: set[tuple[str, str]],
    max_pages_per_province: int = 10,
    min_samples: int = 5,
    sleep_sec: float = 0.2,
) -> pd.DataFrame:
    """시도별로 음식업종 상가업소를 페이지 단위로 훑으며 (시도,시군구)별 좌표 표본을 모은다.

    target_regions에 있는 모든 (SIDO_NM, CCG_NM)이 min_samples 이상 표본을 확보하면
    해당 시도는 조기 종료한다 (API 호출 절약).
    """
    samples: dict[tuple[str, str], list[tuple[float, float]]] = {}

    for sido, ctprvn_cd in SIDO_CTPRVN_CODES.items():
        wanted_in_province = {r for r in target_regions if r[0] == sido}
        if not wanted_in_province:
            continue

        for page in range(1, max_pages_per_province + 1):
            try:
                data = _fetch_page(service_key, ctprvn_cd, page)
            except urllib.error.URLError as e:
                print(f"[경고] {sido} page={page} 호출 실패: {e}")
                break

            if data["header"]["resultMsg"] != "NORMAL SERVICE":
                print(f"[정보] {sido}: {data['header']['resultMsg']} (page={page}에서 중단)")
                break

            items = data["body"].get("items") or []
            if not items:
                break

            for it in items:
                # it["ctprvnNm"]이 아니라 카드데이터 기준 sido를 키로 써야 한다
                # (광주/전남처럼 API상 시도명이 카드데이터와 달라진 경우 매칭이 깨지기 때문).
                key = (sido, it["signguNm"])
                lon, lat = float(it["lon"]), float(it["lat"])
                samples.setdefault(key, []).append((lon, lat))

            covered = {r for r in wanted_in_province if len(samples.get(r, [])) >= min_samples}
            print(f"  {sido} page={page}: 목표지역 커버 {len(covered)}/{len(wanted_in_province)}")
            if covered == wanted_in_province:
                break

            time.sleep(sleep_sec)

    rows = []
    for (sido, ccg), coords in samples.items():
        lons = sorted(c[0] for c in coords)
        lats = sorted(c[1] for c in coords)
        n = len(coords)
        rows.append({
            "SIDO_NM": sido, "CCG_NM": ccg,
            "lon": lons[n // 2], "lat": lats[n // 2],  # 중앙값 -> 이상치(오지번 오류 등)에 덜 민감
            "n_samples": n,
        })
    return pd.DataFrame(rows)


def save_centroids(df: pd.DataFrame, path: Path = CENTROID_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_centroids(path: Path = CENTROID_CACHE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    service_key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
    if not service_key:
        raise SystemExit(
            "환경변수 DATA_GO_KR_SERVICE_KEY가 필요합니다.\n"
            "사용법: DATA_GO_KR_SERVICE_KEY=발급받은키 python src/store_data.py"
        )

    segment_path = PROJECT_ROOT / "results" / "segment_classification.csv"
    segment_df = pd.read_csv(segment_path)
    target_regions = set(zip(segment_df["SIDO_NM"], segment_df["CCG_NM"]))
    print(f"[OK] 목표 지역 수: {len(target_regions)}")

    # 기존 캐시가 있으면 이미 확보한 지역은 재요청하지 않고, 누락된 지역만 추가 수집한다
    existing = load_centroids() if CENTROID_CACHE_PATH.exists() else pd.DataFrame(columns=["SIDO_NM", "CCG_NM", "lon", "lat", "n_samples"])
    already_have = set(zip(existing["SIDO_NM"], existing["CCG_NM"]))
    remaining = target_regions - already_have
    print(f"[OK] 기존 캐시 {len(already_have)}개 재사용, 신규 수집 대상 {len(remaining)}개")

    if remaining:
        new_centroids = fetch_region_centroids(service_key, remaining, max_pages_per_province=15, sleep_sec=0.4)
        centroids = pd.concat([existing, new_centroids], ignore_index=True).drop_duplicates(["SIDO_NM", "CCG_NM"])
    else:
        centroids = existing

    save_centroids(centroids)
    matched = len(set(zip(centroids["SIDO_NM"], centroids["CCG_NM"])) & target_regions)
    print(f"[OK] 저장 완료: {CENTROID_CACHE_PATH} ({len(centroids)}개 지역)")
    print(f"[OK] 목표 255개 지역 중 실좌표 확보: {matched}개 ({matched/len(target_regions)*100:.1f}%)")
