import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. Page configuration
st.set_page_config(
    page_title="Discrete Fourier Series Approximator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Discrete Fourier Series Approximator")

st.markdown(
    r"""
This app approximates periodic functions using discrete Fourier coefficients.

\[
a_0=\frac{1}{N}\sum_{i=1}^{N}f(x_i)
\]

\[
a_n=\frac{2}{N}\sum_{i=1}^{N}f(x_i)\cos(nx_i)
\]

\[
b_n=\frac{2}{N}\sum_{i=1}^{N}f(x_i)\sin(nx_i)
\]

\[
\hat f(x)=a_0+\sum_{n=1}^{M}
\left[a_n\cos(nx)+b_n\sin(nx)\right]
\]
"""
)

st.info(
    "For best results, use uniformly spaced sample points over one "
    "complete period of length 2π, without repeating the endpoint."
)

# 2. Default sample data
if "points" not in st.session_state:
    x_default = np.linspace(0, 2 * np.pi, 9)[:-1]
    y_default = (
        np.sin(x_default)
        + 0.5 * np.cos(2 * x_default)
        + 2
    )

    st.session_state.points = pd.DataFrame({
        "x": x_default,
        "f(x)": y_default,
    })

st.subheader("1. Data Points")

edited_data = st.data_editor(
    st.session_state.points,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "x": st.column_config.NumberColumn(
            "x (radians)", format="%.4f"
        ),
        "f(x)": st.column_config.NumberColumn(
            "f(x)", format="%.4f"
        ),
    },
    key="data_editor",
)

# 3. Validate and clean data
clean_data = edited_data.copy()
clean_data["x"] = pd.to_numeric(
    clean_data["x"], errors="coerce"
)
clean_data["f(x)"] = pd.to_numeric(
    clean_data["f(x)"], errors="coerce"
)
clean_data = clean_data.dropna(subset=["x", "f(x)"])

N_points = len(clean_data)

# 4. Harmonic selection
st.subheader("2. Harmonics")

max_harmonic = st.slider(
    "Select M (Harmonics Order)",
    min_value=1,
    max_value=20,
    value=4,
    step=1,
)

# 5. Sampling warning
if N_points >= 3 and max_harmonic > N_points // 2:
    st.warning(
        f"With {N_points} sample points, high harmonic orders "
        "may be affected by aliasing and sampling limitations. "
        "Consider increasing the number of uniformly spaced "
        "sample points. This is a caution, not a guarantee "
        "that the approximation is invalid."
    )

calculate_button = st.button(
    "Calculate Fourier Series",
    type="primary",
    use_container_width=True,
)


# 6. Fourier coefficient functions
def calculate_coefficients(x_values, y_values, harmonics):
    N = len(y_values)
    a0 = np.mean(y_values)

    rows = []

    for n in range(1, harmonics + 1):
        an = (2.0 / N) * np.sum(
            y_values * np.cos(n * x_values)
        )
        bn = (2.0 / N) * np.sum(
            y_values * np.sin(n * x_values)
        )

        rows.append({
            "n": n,
            "an": an,
            "bn": bn,
        })

    return a0, pd.DataFrame(
        rows, columns=["n", "an", "bn"]
    )


def evaluate_fourier_series(x_values, a0, coefficients):
    result = np.full_like(
        np.asarray(x_values, dtype=float),
        a0,
        dtype=float,
    )

    for row in coefficients.itertuples(index=False):
        result += (
            row.an * np.cos(row.n * x_values)
            + row.bn * np.sin(row.n * x_values)
        )

    return result


# 7. Perform calculation
if calculate_button:
    if len(clean_data) < 3:
        st.error("Please enter at least 3 valid points.")
        st.stop()

    if clean_data["x"].duplicated().any():
        st.error("x values must be unique.")
        st.stop()

    data = clean_data.sort_values("x")
    x_samples = data["x"].to_numpy(dtype=float)
    y_samples = data["f(x)"].to_numpy(dtype=float)

    if not np.all(np.isfinite(x_samples)) or not np.all(
        np.isfinite(y_samples)
    ):
        st.error("All input values must be finite numbers.")
        st.stop()

    # Check whether sample points are uniformly spaced
    diffs = np.diff(x_samples)
    if len(diffs) > 1 and not np.allclose(
        diffs, diffs[0], rtol=1e-3, atol=1e-8
    ):
        st.warning(
            "Sample points are not evenly spaced. "
            "The discrete coefficient formulas assume "
            "uniform sampling, so results may be inaccurate."
        )

    # Coefficients
    a0, coefficient_table = calculate_coefficients(
        x_samples,
        y_samples,
        int(max_harmonic),
    )

    # Smooth approximation curve
    x_plot = np.linspace(
        x_samples.min(),
        x_samples.max(),
        1000,
    )

    y_plot = evaluate_fourier_series(
        x_plot, a0, coefficient_table
    )

    y_estimated = evaluate_fourier_series(
        x_samples, a0, coefficient_table
    )

    mse = np.mean((y_samples - y_estimated) ** 2)

    # 8. Results
    st.divider()
    st.subheader("3. Calculation Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("N (Sample Points)", N_points)
    col2.metric("M (Harmonics)", int(max_harmonic))
    col3.metric("MSE", f"{mse:.6f}")

    st.latex(rf"a_0 = {a0:.6f}")

    # 9. Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        mode="lines",
        name=f"Fourier Approximation (M={max_harmonic})",
        line={"color": "#1f77b4", "width": 3},
    ))

    fig.add_trace(go.Scatter(
        x=x_samples,
        y=y_samples,
        mode="markers",
        name="Input Points",
        marker={
            "color": "#d62728",
            "size": 11,
            "line": {"color": "white", "width": 1},
        },
    ))

    fig.update_layout(
        title="Input Data vs Fourier Approximation",
        xaxis_title="x (radians)",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # 10. Approximation formula
    st.subheader("4. Approximation Formula")

    formula = f"\\hat{{f}}(x) = {a0:.4f}"

    for row in coefficient_table.itertuples(index=False):
        if abs(row.an) > 1e-4:
            sign = "+" if row.an >= 0 else "-"
            formula += (
                f" {sign} {abs(row.an):.4f}"
                f"\\cos({row.n}x)"
            )

        if abs(row.bn) > 1e-4:
            sign = "+" if row.bn >= 0 else "-"
            formula += (
                f" {sign} {abs(row.bn):.4f}"
                f"\\sin({row.n}x)"
            )

    st.latex(formula)

    # 11. Coefficient table
    st.subheader("5. Fourier Coefficients")

    with st.expander("Show Coefficients Table", expanded=True):
        st.dataframe(
            coefficient_table,
            use_container_width=True,
            hide_index=True,
        )

    # 12. CSV download
    csv_data = coefficient_table.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Coefficients as CSV",
        data=csv_data,
        file_name="fourier_coefficients.csv",
        mime="text/csv",
    )