import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Case 8 – State-Space Detective Lab",
    page_icon="🕵️‍♂️",
)

st.title("🕵️‍♂️ Case 8: The Sliding Car’s Secret Blueprint")
st.write(
    """
In this lab you revisit the **sliding car** – now with a full **state-space model**.

- State: **x** = position, **v** = velocity  
- Input: **u** = control force (0–100 %)  
- Model: \\(X_{k+1} = A_d X_k + B_d u_k\\)  
- Controller: state feedback

You will:

1. Tune the physical parameters (mass, friction).  
2. See how they define the discrete-time matrices **A_d** and **B_d**.  
3. Choose state-feedback gains **k_x** and **k_v**.  
4. Observe the closed-loop response and eigenvalues.
"""
)

st.markdown("---")

# -----------------------------
# 1) Physical system parameters
# -----------------------------
st.subheader("1️⃣ Physical System – Sliding Car")

col_sys1, col_sys2, col_sys3 = st.columns(3)
with col_sys1:
    m = st.slider(
        "Mass m (kg)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Heavier carts react more slowly to the same force.",
    )
with col_sys2:
    c = st.slider(
        "Friction coefficient c",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Larger c means stronger velocity damping.",
    )
with col_sys3:
    x_ref = st.slider(
        "Target position x_ref (m)",
        min_value=-2.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )

st.write(
    f"Selected physical parameters: **m = {m:.1f} kg**, "
    f"**c = {c:.2f}**, target **x_ref = {x_ref:.2f} m**"
)

# -----------------------------
# 2) Simulation settings
# -----------------------------
st.subheader("2️⃣ Simulation Settings")

col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    t_max = st.slider(
        "Total time (s)",
        min_value=2.0,
        max_value=20.0,
        value=8.0,
        step=1.0,
    )
with col_sim2:
    dt = st.slider(
        "Time step Δt (s)",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
    )

n_steps = int(t_max / dt) + 1
st.write(f"Simulation: **{t_max:.1f} s** with Δt = **{dt:.3f} s** "
         f"(≈ {n_steps} steps)")

col_init1, col_init2 = st.columns(2)
with col_init1:
    x0 = st.slider(
        "Initial position x₀ (m)",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )
with col_init2:
    v0 = st.slider(
        "Initial velocity v₀ (m/s)",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )

# -----------------------------
# 3) Controller parameters
# -----------------------------
st.subheader("3️⃣ State-Feedback Controller Gains")

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    kx = st.slider(
        "Position gain k_x",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )
with col_ctrl2:
    kv = st.slider(
        "Velocity gain k_v",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

st.caption(
    "Control law:  u = -k_x (x - x_ref) - k_v v.  "
    "High k_x reacts strongly to position error; "
    "high k_v damps velocity."
)

# -----------------------------
# 4) Build discrete-time model
# -----------------------------
st.subheader("4️⃣ Discrete-Time State-Space Model")

# A_d and B_d from simple Euler discretization of the sliding cart
A11 = 1.0
A12 = dt
A21 = 0.0
A22 = 1.0 - (c / m) * dt
B1 = 0.0
B2 = dt / m

A_d = np.array([[A11, A12],
                [A21, A22]])
B_d = np.array([[B1],
                [B2]])

# Latex display of A_d and B_d
A_tex = (
    r"A_d = \begin{{bmatrix}} {a11:.3f} & {a12:.3f} \\ 0 & {a22:.3f} \end{{bmatrix}}"
    .format(a11=A11, a12=A12, a22=A22)
)
B_tex = (
    r"B_d = \begin{{bmatrix}} {b1:.3f} \\ {b2:.3f} \end{{bmatrix}}"
    .format(b1=B1, b2=B2)
)

st.latex(A_tex)
st.latex(B_tex)

st.caption(
    r"Model: $X_{k+1} = A_d X_k + B_d u_k$ with "
    r"$X_k = \begin{bmatrix} x_k \\ v_k \end{bmatrix}$."
)

# -----------------------------
# 5) Simulation functions
# -----------------------------
def simulate_cart(
    A_d, B_d, kx, kv, x_ref, x0, v0, dt, n_steps
):
    """Simulate closed-loop sliding cart."""
    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    u = np.zeros(n_steps)

    x[0] = x0
    v[0] = v0

    for k in range(n_steps - 1):
        # error relative to reference
        e_pos = x[k] - x_ref
        e_vel = v[k]

        # state-feedback control law
        u[k] = -kx * e_pos - kv * e_vel
        # clip control to a reasonable range (0–100 % equivalent)
        u[k] = np.clip(u[k], -100.0, 100.0)

        X_k = np.array([[x[k]], [v[k]]])
        X_next = A_d @ X_k + B_d * u[k]

        x[k + 1] = X_next[0, 0]
        v[k + 1] = X_next[1, 0]
        t[k + 1] = t[k] + dt

    u[-1] = u[-2]
    return t, x, v, u


def format_complex(z: complex) -> str:
    if abs(z.imag) < 1e-6:
        return f"{z.real:.3f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.3f} {sign} {abs(z.imag):.3f}j"


# -----------------------------
# 6) Run simulation
# -----------------------------
t, x, v, u = simulate_cart(
    A_d, B_d, kx, kv, x_ref, x0, v0, dt, n_steps
)

# Eigenvalues (open-loop and closed-loop)
eig_open, _ = np.linalg.eig(A_d)
K = np.array([[kx, kv]])  # 1x2
A_cl = A_d - B_d @ K      # closed-loop matrix
eig_closed, _ = np.linalg.eig(A_cl)

st.markdown("---")
st.subheader("5️⃣ Time-Domain Response")

# Position and setpoint
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(t, x, label="Position x(t)")
ax1.axhline(x_ref, color="gray", linestyle="--", label="Setpoint")
ax1.set_xlabel("t (s)")
ax1.set_ylabel("x (m)")
ax1.set_title("Cart Position vs Time")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()
st.pyplot(fig1)

# Velocity
fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(t, v, label="Velocity v(t)")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("v (m/s)")
ax2.set_title("Cart Velocity vs Time")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()
st.pyplot(fig2)

# Control signal
fig3, ax3 = plt.subplots(figsize=(7, 3))
ax3.plot(t, u, label="Control u(t)")
ax3.set_xlabel("t (s)")
ax3.set_ylabel("u (force units)")
ax3.set_title("Control Effort")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend()
st.pyplot(fig3)

# -----------------------------
# 7) Eigenvalue radar
# -----------------------------
st.subheader("6️⃣ Eigenvalue Radar – Open vs Closed Loop")

fig4, ax4 = plt.subplots(figsize=(4, 4))

# Unit circle for discrete-time stability
theta = np.linspace(0, 2 * np.pi, 200)
ax4.plot(np.cos(theta), np.sin(theta), linestyle="--", alpha=0.5, label="Unit circle")

# Open-loop eigenvalues
ax4.scatter(eig_open.real, eig_open.imag, marker="o", label="Open-loop eig")
# Closed-loop eigenvalues
ax4.scatter(eig_closed.real, eig_closed.imag, marker="x", s=70, label="Closed-loop eig")

ax4.axhline(0, color="black", linewidth=0.5)
ax4.axvline(0, color="black", linewidth=0.5)
ax4.set_xlabel("Real part")
ax4.set_ylabel("Imag part")
ax4.set_title("Eigenvalues in the Complex Plane")
ax4.set_aspect("equal", "box")
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend()
st.pyplot(fig4)

st.write(
    "**Open-loop eigenvalues:** "
    + ", ".join(format_complex(z) for z in eig_open)
)
st.write(
    "**Closed-loop eigenvalues:** "
    + ", ".join(format_complex(z) for z in eig_closed)
)

# -----------------------------
# 8) Teacher / Discussion box
# -----------------------------
st.markdown("---")
with st.expander("👩‍🏫 Teacher Box – State-Space & Feedback Intuition"):
    st.write(
        r"""
- We model the cart with a **2D state**: position and velocity.
- The matrices $A_d$ and $B_d$ describe **how the state moves forward**
  in one time step when a control $u_k$ is applied.
- State feedback $u_k = -k_x (x_k - x_\text{ref}) - k_v v_k$ is like giving
  the controller **direct access** to the internal state.

Key discussion points:

1. How do increases in $k_x$ and $k_v$ move the closed-loop eigenvalues?
2. What patterns do you see when the eigenvalues
   - are real and near $0$,
   - are complex but inside the unit circle,
   - cross outside the unit circle?
3. Compare the time-domain plots (position, velocity, control) with
   the eigenvalue radar: how do they tell the **same story** in two languages?
"""
    )

st.caption(
    "Case 8 – State-Space Detective Lab: A bridge from high-school PID "
    "to early university state-space and eigenvalue analysis."
)
