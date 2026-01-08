import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pipeline_v as plv
from branca.colormap import linear
import folium
from folium.plugins import HeatMap



def plot_distribution_prix_m2_appartements(
    df: pd.DataFrame,
    postalCode: str | None = None,
    postal_col: str = "computedPostalCode",
    col_type: str = "propertyType",
    valeurs_appart: tuple = ("flat", "apartment", "Appartement", "appartement"),
    col_prix_m2: str = "pricePerSquareMeter",
    col_prix: str = "price",
    col_surface: str = "surfaceArea",
    bins: int = 25,
    clip_quantiles: tuple | None = (0.01, 0.99),
):
    d = df.copy()
    if postalCode is not None and str(postalCode).lower() != "tous":
        if postal_col not in d.columns:
            raise ValueError(f"Colonne code postal introuvable: {postal_col}")
        d = d[d[postal_col].astype(str) == str(postalCode)]

    if col_type in d.columns:
        d = d[d[col_type].isin(valeurs_appart)]

    if col_prix_m2 in d.columns:
        x = pd.to_numeric(d[col_prix_m2], errors="coerce")
    else:
        x = pd.to_numeric(d[col_prix], errors="coerce") / pd.to_numeric(
            d[col_surface], errors="coerce"
        )

    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if clip_quantiles is not None and len(x) > 10:
        q_low, q_high = x.quantile(clip_quantiles[0]), x.quantile(clip_quantiles[1])
        x = x[(x >= q_low) & (x <= q_high)]

    if len(x) < 3:
        raise ValueError("Pas assez de valeurs de prix/m2 apres nettoyage/filtrage.")

    mu = float(x.mean())
    sigma = float(x.std(ddof=1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(x, bins=bins, density=True, alpha=0.7)

    xs = np.linspace(x.min(), x.max(), 400)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((xs - mu) / sigma) ** 2
    )
    ax.plot(xs, pdf, linewidth=2)

    ax.set_title(
        "Distribution du prix/m2 (appartements)\n"
        f"Moyenne = {mu:.2f} | Ecart-type = {sigma:.2f} | Nbr d'annonces = {len(x)}"
    )
    ax.set_xlabel("Prix / m2")
    ax.set_ylabel("Densite")
    ax.grid(True, alpha=0.2)

    return fig, mu, sigma, len(x)
def plot_scatter_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "price",
    sample_max: int | None = None,
    clip_q: tuple | None = (0.01, 0.99),
): 
    if x_col not in df.columns:
        raise ValueError(f"Colonne X introuvable: {x_col}")
    if y_col not in df.columns:
        raise ValueError(f"Colonne Y introuvable: {y_col}")

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    d = pd.DataFrame({"x": x[mask], "y": y[mask]})

    if len(d) < 3:
        raise ValueError("Pas assez de donnees valides pour faire une regression.")

    if clip_q is not None and len(d) > 20:
        x_low, x_high = np.quantile(d["x"], clip_q)
        y_low, y_high = np.quantile(d["y"], clip_q)
        d = d[(d["x"] >= x_low) & (d["x"] <= x_high) & (d["y"] >= y_low) & (d["y"] <= y_high)]

    g = (
        d.groupby("x", as_index=False)
        .agg(y_mean=("y", "mean"), n=("y", "size"))
        .sort_values("x")
    )

    if len(g) < 3:
        raise ValueError("Pas assez de points agreges pour faire une regression.")

    if sample_max is not None and len(g) > sample_max:
        idx = np.random.choice(len(g), size=sample_max, replace=False)
        g = g.iloc[idx].sort_values("x")

    a, b = np.polyfit(g["x"], g["y_mean"], 1)
    y_hat = a * g["x"] + b

    ss_res = np.sum((g["y_mean"] - y_hat) ** 2)
    ss_tot = np.sum((g["y_mean"] - np.mean(g["y_mean"])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    xs = np.linspace(g["x"].min(), g["x"].max(), 200)
    ys = a * xs + b

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(g["x"], g["y_mean"], c=g["n"], cmap="viridis", s=35, alpha=0.85)
    ax.plot(xs, ys, linewidth=2,color = "green")
    cbar = fig.colorbar(sc)
    cbar.set_label("Densite (nb d'annonces)")

    ax.set_title(
        f"Regression lineaire (moyenne par valeur de X) : {y_col} ~ {x_col}\n"
        f"y = {a:.3f}x + {b:.3f} | R2 = {r2:.3f} | n = {len(d)} | points = {len(g)}"
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.2)

    return fig, a, b, r2, len(g)


def carte_prix_couleur_densite_taille(
    df: pd.DataFrame,
    postal_col: str = "postalCode",
    lat_col: str = "blurInfo.centroid.lat",
    lon_col: str = "blurInfo.centroid.lon",
    price_m2_col: str = "pricePerSquareMeter",
    agg_price: str = "mean",  # "mean" ou "median"
    min_count: int = 3,  # ignore les CP avec trop peu d'annonces
    add_density_heatmap: bool = True,  # couche optionnelle "densité brute" (tous les points)
    radius_min: int = 4,
    radius_max: int = 18,
):
    d = df[[postal_col, lat_col, lon_col, price_m2_col]].copy()
    d[postal_col] = pd.to_numeric(d[postal_col], errors="coerce")
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d[price_m2_col] = pd.to_numeric(d[price_m2_col], errors="coerce")
    d = d.dropna()

    if len(d) == 0:
        raise ValueError("Aucune donnée exploitable (NaN partout après nettoyage).")
    f_agg = np.mean if agg_price == "mean" else np.median
    g = (
        d.groupby(postal_col)
        .agg(
            lat=(lat_col, "mean"),
            lon=(lon_col, "mean"),
            price_m2=(price_m2_col, f_agg),
            n=(postal_col, "size"),
        )
        .reset_index()
    )

    g = g[g["n"] >= min_count].copy()
    if len(g) == 0:
        raise ValueError("Aucun code postal après filtrage min_count. Diminue min_count.")

    # --- Carte de base ---
    m = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles="CartoDB positron")

    # --- Colormap (prix/m²) ---
    vmin = float(g["price_m2"].quantile(0.02))
    vmax = float(g["price_m2"].quantile(0.98))
    if vmin == vmax:
        vmin = float(g["price_m2"].min())
        vmax = float(g["price_m2"].max()) + 1e-9

    colormap = linear.plasma.scale(vmin, vmax)
    colormap.caption = f"{agg_price} {price_m2_col} (€/m² ou unité dataset)"
    colormap.add_to(m)
    n_min, n_max = int(g["n"].min()), int(g["n"].max())

    def scale_radius(n):
        if n_max == n_min:
            return (radius_min + radius_max) / 2
        t = (np.sqrt(n) - np.sqrt(n_min)) / (np.sqrt(n_max) - np.sqrt(n_min))
        return radius_min + t * (radius_max - radius_min)

    fg = folium.FeatureGroup(name="Prix/m² (couleur) + Densité (taille)", show=True)

    for _, row in g.iterrows():
        cp = int(row[postal_col])
        lat, lon = float(row["lat"]), float(row["lon"])
        pm2 = float(row["price_m2"])
        n = int(row["n"])

        color = colormap(pm2)

        folium.CircleMarker(
            location=[lat, lon],
            radius=scale_radius(n),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            popup=(
                f"CP {cp}<br>"
                f"{agg_price} prix/m² = {pm2:.2f}<br>"
                f"Annonces = {n}"
            ),
        ).add_to(fg)

    fg.add_to(m)

    if add_density_heatmap:
        fg2 = folium.FeatureGroup(name="Densité brute (HeatMap)", show=False)
        heat_points = d[[lat_col, lon_col]].values.tolist()
        HeatMap(heat_points, radius=12, blur=18, min_opacity=0.25).add_to(fg2)
        fg2.add_to(m)

    ex = sorted(set([n_min, int(np.median(g["n"])), n_max]))

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border: 1px solid #ccc;
        border-radius: 6px;
        font-size: 12px;
        ">
        <b>Densité (taille)</b><br>
        <span>nb d'annonces (par CP)</span><br><br>
        {''.join([f'''
        <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
            <div style="
                width:{2*scale_radius(v):.0f}px; height:{2*scale_radius(v):.0f}px;
                border:2px solid #555; border-radius:50%;
                background: rgba(0,0,0,0.08);
            "></div>
            <div>{v}</div>
        </div>''' for v in ex])}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m

