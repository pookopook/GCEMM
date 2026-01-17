import numpy as np
import pandas as pd
def test_sum(a:int=1,b:int=2):
    return a+b


def simulate_corsia_mrv_fuel_burn(
    n_airlines: int = 600,
    years=range(2019, 2026),
    seed: int = 42,

    # Fuel burn 분포(heavy-tail): lognormal(mean=ln_mu, sigma=ln_sigma)
    # exp(ln_mu)는 2019년 중앙값 근처의 연료소모량(ton fuel)
    ln_mu: float = 12.8,            # exp(12.8) ≈ 363k ton fuel (중앙값 느낌)
    ln_sigma: float = 1.20,         # 쏠림 정도 (1.0~1.5면 강한 상위 집중)

    # MRV 대상처럼 "너무 작은 사업자" 제거(연간 최소 연료소모량)
    mrv_min_annual_tfuel: float = 80_000,

    # 항공사별 연평균 성장률(연료소모 성장)
    g_mean: float = 0.03,
    g_sd: float = 0.04,

    # 연도 공통 충격(글로벌 수요/유가/운항환경)
    shock_sd: float = 0.08,

    # 항공사-연도 특이 변동(소규모 noise)
    idio_sd: float = 0.05,
):
    rng = np.random.default_rng(seed)

    # 간단 국가/지역 더미(항공사 국적 = AOC 발급국)
    countries = [
        ("United States", "NAM"), ("China", "APAC"), ("United Kingdom", "EUR"),
        ("Germany", "EUR"), ("France", "EUR"), ("United Arab Emirates", "MEA"),
        ("Qatar", "MEA"), ("Turkey", "EUR"), ("Singapore", "APAC"),
        ("Japan", "APAC"), ("Korea, Rep.", "APAC"), ("Australia", "APAC"),
        ("India", "APAC"), ("Ethiopia", "AFR"), ("South Africa", "AFR"),
        ("Brazil", "LAM"), ("Mexico", "LAM"), ("Chile", "LAM"),
        ("Kenya", "AFR"), ("Canada", "NAM")
    ]
    country_names = np.array([c[0] for c in countries])
    country_regions = {c[0]: c[1] for c in countries}

    # 국가 가중치(대형 시장에 항공사 더 배치되는 느낌)
    weights = np.array([
        0.16, 0.14, 0.06, 0.06, 0.05, 0.05,
        0.04, 0.04, 0.04, 0.04, 0.04, 0.03,
        0.06, 0.02, 0.02, 0.04, 0.03, 0.02,
        0.02, 0.04
    ])
    weights = weights / weights.sum()

    # 항공사 식별자 + 국적 부여
    airline_ids = [f"AL{str(i).zfill(4)}" for i in range(1, n_airlines + 1)]
    operator_state = rng.choice(country_names, size=n_airlines, p=weights)
    region = np.array([country_regions[s] for s in operator_state])

    # 2019 기준 "기본 연료소모" (heavy-tail)
    base_fuel_2019 = rng.lognormal(mean=ln_mu, sigma=ln_sigma, size=n_airlines)

    # MRV 대상처럼 최소 컷 적용 (부족하면 추가 샘플로 채움)
    mask = base_fuel_2019 >= mrv_min_annual_tfuel
    while mask.sum() < n_airlines:
        extra = rng.lognormal(mean=ln_mu, sigma=ln_sigma, size=n_airlines)
        base_fuel_2019 = np.where(mask, base_fuel_2019, extra)
        mask = base_fuel_2019 >= mrv_min_annual_tfuel

    # 항공사별 성장률
    g_i = rng.normal(loc=g_mean, scale=g_sd, size=n_airlines)
    g_i = np.clip(g_i, -0.10, 0.20)

    # 연도별 공통 충격(로그 스케일)
    years = list(years)
    year_shocks = {y: rng.normal(loc=0.0, scale=shock_sd) for y in years}

    rows = []
    for idx in range(n_airlines):
        al_id = airline_ids[idx]
        st = operator_state[idx]
        reg = region[idx]
        f0 = base_fuel_2019[idx]
        gi = g_i[idx]

        for y in years:
            t = y - 2019
            eps = rng.normal(0.0, idio_sd)

            # FuelBurn = f0 * (1+gi)^t * exp(year_shock) * exp(idio_noise)
            fuel = f0 * ((1.0 + gi) ** t) * np.exp(year_shocks[y]) * np.exp(eps)

            rows.append({
                "airline_id": al_id,
                "operator_state": st,
                "region": reg,
                "year": y,
                "fuel_burn_tonnes": float(fuel),
                "g_i": float(gi),
                "year_shock": float(year_shocks[y])
            })

    df = pd.DataFrame(rows).sort_values(["airline_id", "year"]).reset_index(drop=True)

    # 연도별 점유율(상위 집중 확인용)
    df["fuel_share_in_year"] = df.groupby("year")["fuel_burn_tonnes"].apply(lambda s: s / s.sum()).reset_index(level=0, drop=True)

    return df



if __name__ == "__main__":
    df_fuel = simulate_corsia_mrv_fuel_burn(
        n_airlines=600,
        years=range(2019, 2026),
        seed=7,
        ln_mu=12.8,
        ln_sigma=1.25,
        mrv_min_annual_tfuel=100_000
    )

    print(df_fuel.head(10))

    print("\nYearly totals (million tonnes fuel):")
    print((df_fuel.groupby("year")["fuel_burn_tonnes"].sum() / 1e6).round(2))

    top10_share = (
        df_fuel.sort_values(["year", "fuel_burn_tonnes"], ascending=[True, False])
               .groupby("year").head(10)
               .groupby("year")["fuel_burn_tonnes"].sum()
        / df_fuel.groupby("year")["fuel_burn_tonnes"].sum()
    )
    print("\nTop-10 airline fuel share by year:")
    print(top10_share.round(3))

# df_fuel[df_fuel['year'] == 2019].to_csv('./dataset/fuel_burn_2019_랜덤생성.csv', index=False)