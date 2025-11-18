# app.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # importante para Streamlit
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from keras.models import load_model
import pickle


# ==============================
# Parámetros generales
# ==============================
TARGET_COL = "pm2.5"
SEQ_LENGTH_DEFAULT = 6
N_FUTURE_DEFAULT = 6
N_PAST_PRED_DEFAULT = 6


# ==============================
# Funciones auxiliares
# ==============================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])   # todas las features
        y.append(data[i + seq_length, 0])     # solo el target
    return np.array(X), np.array(y)


@st.cache_resource
def load_trained_model():
    try:
        model = load_model("gru_pm25_model.keras")
    except Exception:
        model = load_model("gru_pm25_model.h5")
    return model


@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler


@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("contamination.csv")

    # Crear DateTime e indice
    df["DateTime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.set_index("DateTime", inplace=True)

    # Eliminar columnas que no usa el modelo
    df = df.drop(["year", "month", "day", "hour", "No"], axis=1, errors="ignore")

    # Rellenar faltantes
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Orden de columnas: target primero + resto de features
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    data = df[[TARGET_COL] + feature_cols].values

    return df, data, feature_cols


def prepare_sequences_for_eval(data, scaler, seq_length):
    # Normalizar usando el scaler entrenado
    data_scaled = scaler.transform(data)

    # Crear secuencias
    X, y = create_sequences(data_scaled, seq_length)

    # Split sin mezclar, como en el script original
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, data_scaled


def denormalize_target(vec_scaled, scaler, n_features):
    """
    Recibe un vector 1D de valores escalados del target
    y lo desnormaliza usando el mismo scaler (MinMax).
    """
    vec_scaled = np.asarray(vec_scaled).reshape(-1, 1)
    zeros = np.zeros((len(vec_scaled), n_features - 1))
    tmp = np.column_stack([vec_scaled, zeros])
    denorm = scaler.inverse_transform(tmp)[:, 0]
    return denorm


# ==============================
# Streamlit UI
# ==============================
st.set_page_config(
    page_title="GRU PM2.5 Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("Pronóstico de PM2.5 con modelo GRU")

st.markdown(
    """
Aplicación web para evaluar y visualizar tu modelo de series de tiempo GRU 
entrenado para predecir la concentración de PM2.5.
"""
)

# Sidebar
st.sidebar.header("Parámetros")
seq_length = 6

n_future = st.sidebar.number_input(
    "Horas futuras a pronosticar",
    min_value=1,
    max_value=24,
    value=N_FUTURE_DEFAULT,
    step=1
)
n_past_pred = st.sidebar.number_input(
    "Horas pasadas a recalcular (para -h a 0)",
    min_value=1,
    max_value=24,
    value=N_PAST_PRED_DEFAULT,
    step=1
)
n_mc_iter = st.sidebar.number_input(
    "Iteraciones Monte Carlo (cono de incertidumbre)",
    min_value=10,
    max_value=300,
    value=100,
    step=10
)

# ==============================
# Carga de modelo, scaler y datos
# ==============================
with st.spinner("Cargando modelo, scaler y datos..."):
    model = load_trained_model()
    scaler = load_scaler()
    df, data, feature_cols = load_and_prepare_data()
    X_train, X_test, y_train, y_test, data_scaled = prepare_sequences_for_eval(
        data, scaler, seq_length
    )

n_features = data.shape[1]

# ==============================
# Predicciones y métricas
# ==============================
y_train_pred_scaled = model.predict(X_train, verbose=0)
y_test_pred_scaled = model.predict(X_test, verbose=0)

# Desnormalizar
y_train_actual = denormalize_target(y_train, scaler, n_features)
y_test_actual = denormalize_target(y_test, scaler, n_features)
y_train_pred_actual = denormalize_target(y_train_pred_scaled.flatten(), scaler, n_features)
y_test_pred_actual = denormalize_target(y_test_pred_scaled.flatten(), scaler, n_features)

# Métricas
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
train_r2 = r2_score(y_train_actual, y_train_pred_actual)
test_r2 = r2_score(y_test_actual, y_test_pred_actual)
train_mae = np.mean(np.abs(y_train_actual - y_train_pred_actual))
test_mae = np.mean(np.abs(y_test_actual - y_test_pred_actual))

st.subheader("Métricas del modelo (escala original de PM2.5)")
col1, col2 = st.columns(2)
with col1:
    st.write(f"RMSE entrenamiento: {train_rmse:.2f}")
    st.write(f"MAE entrenamiento: {train_mae:.2f}")
    st.write(f"R² entrenamiento: {train_r2:.3f}")
with col2:
    st.write(f"RMSE prueba: {test_rmse:.2f}")
    st.write(f"MAE prueba: {test_mae:.2f}")
    st.write(f"R² prueba: {test_r2:.3f}")

st.markdown("---")

# ==============================
# Gráfica 1: Predicciones vs reales (test)
# ==============================
st.subheader("Predicciones vs valores reales (conjunto de prueba)")

sample_size = min(500, len(y_test_actual))
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_test_actual[:sample_size], label="Real", linewidth=1.5)
ax1.plot(y_test_pred_actual[:sample_size], label="Predicción", linewidth=1.5)
ax1.set_title(f"Test: real vs predicción (primeras {sample_size} muestras)")
ax1.set_xlabel("Muestra")
ax1.set_ylabel("PM2.5")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# ==============================
# Gráfica 2: Dispersión Real vs Predicho (test)
# ==============================
st.subheader("Dispersión valores reales vs predichos (test)")

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.scatter(y_test_actual, y_test_pred_actual, alpha=0.5, s=10)
min_val = min(y_test_actual.min(), y_test_pred_actual.min())
max_val = max(y_test_actual.max(), y_test_pred_actual.max())
ax2.plot([min_val, max_val], [min_val, max_val], "r--", label="Línea perfecta")
ax2.set_xlabel("Real")
ax2.set_ylabel("Predicho")
ax2.set_title("Predicho vs Real (test)")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.markdown("---")

# ==============================
# Gráfica 3: Test data vs prediction en el tiempo (días)
# ==============================
st.subheader("Serie temporal: valores reales vs predicción (test, en días)")

# Construir eje de tiempo en días (asumiendo datos horarios)
days_since_start = np.arange(len(y_test_actual)) / 24.0

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(days_since_start, y_test_actual, label="Real", linewidth=1.5)
ax3.plot(days_since_start, y_test_pred_actual, label="Predicción", linewidth=1.5)
ax3.set_xlabel("Tiempo (días)")
ax3.set_ylabel("PM2.5")
ax3.set_title("Datos de prueba vs predicción (GRU)")
ax3.legend()
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

st.markdown("---")

# ==============================
# Gráfica 4: Últimas 24 h y pronóstico futuro
# ==============================
st.subheader("Últimas 24 horas y pronóstico de las siguientes horas")

# Última secuencia completa para pronóstico futuro
last_sequence_full = data_scaled[-seq_length:, :].reshape(1, seq_length, -1)

future_predictions_scaled = []
current_sequence = last_sequence_full.copy()

for i in range(n_future):
    next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
    future_predictions_scaled.append(next_pred_scaled)

    new_row = current_sequence[0, -1, :].copy().reshape(1, 1, -1)
    new_row[0, 0, 0] = next_pred_scaled
    current_sequence = np.concatenate(
        [current_sequence[:, 1:, :], new_row],
        axis=1
    )

future_predictions = denormalize_target(future_predictions_scaled, scaler, n_features)

# Recalcular predicciones para las últimas n_past_pred horas
last_idx = len(data_scaled) - 1
target_indices_past = range(last_idx - n_past_pred, last_idx)

X_past = []
for i in target_indices_past:
    start = i - seq_length
    end = i
    X_past.append(data_scaled[start:end, :])

X_past = np.array(X_past)
past_preds_scaled = model.predict(X_past, verbose=0).flatten()
past_predictions = denormalize_target(past_preds_scaled, scaler, n_features)

# Últimas 24 horas reales
last_24_real = data[-24:, 0]

# Ejes de tiempo
time_history = np.arange(-24, 0)
time_past_pred = np.arange(-n_past_pred, 0)
time_future_pred = np.arange(0, n_future)
time_pred_all = np.concatenate([time_past_pred, time_future_pred])
pred_all = np.concatenate([past_predictions, future_predictions])

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(time_history, last_24_real, label="Histórico (últimas 24 h)", marker="o")
ax4.plot(time_pred_all, pred_all, label="Pronóstico (-h a +h)", marker="s")
ax4.axvline(x=0, color="gray", linestyle="--", label="Inicio pronóstico")
ax4.set_xlabel("Horas relativas (0 = ahora)")
ax4.set_ylabel("PM2.5")
ax4.set_title("Histórico 24 h + pronóstico próximo")
ax4.legend()
ax4.grid(True, alpha=0.3)
st.pyplot(fig4)

st.markdown("---")

# ==============================
# Gráfica 5: Cono de incertidumbre (Monte Carlo Dropout)
# ==============================
st.subheader("Cono de incertidumbre para las próximas horas (Monte Carlo Dropout)")

future_predictions_mc = []

for _ in range(n_mc_iter):
    current_sequence_mc = last_sequence_full.copy()
    preds_iter = []

    for _ in range(n_future):
        # training=True activa dropout en inferencia
        next_pred_scaled = model(current_sequence_mc, training=True).numpy()[0, 0]
        preds_iter.append(next_pred_scaled)

        new_row = current_sequence_mc[0, -1, :].copy().reshape(1, 1, -1)
        new_row[0, 0, 0] = next_pred_scaled
        current_sequence_mc = np.concatenate(
            [current_sequence_mc[:, 1:, :], new_row],
            axis=1
        )

    future_predictions_mc.append(preds_iter)

future_predictions_mc = np.array(future_predictions_mc)

# Desnormalizar cada iteración
future_predictions_mc_denorm = []
for iter_preds in future_predictions_mc:
    denorm_iter = denormalize_target(iter_preds, scaler, n_features)
    future_predictions_mc_denorm.append(denorm_iter)

future_predictions_mc_denorm = np.array(future_predictions_mc_denorm)

mean_prediction = np.mean(future_predictions_mc_denorm, axis=0)
lower_bound = np.percentile(future_predictions_mc_denorm, 5, axis=0)
upper_bound = np.percentile(future_predictions_mc_denorm, 95, axis=0)

fig5, ax5 = plt.subplots(figsize=(10, 4))
ax5.plot(time_history, last_24_real, label="Histórico (24 h)", marker="o")
ax5.plot(time_pred_all, pred_all, label="Pronóstico (-h a +h)", linestyle="--")
ax5.plot(time_future_pred, mean_prediction, label="Media pronóstico (MC)", marker="s")
ax5.fill_between(
    time_future_pred,
    lower_bound,
    upper_bound,
    alpha=0.3,
    label="Incertidumbre (IC 90%)"
)
ax5.axvline(x=0, color="gray", linestyle="--", label="Inicio pronóstico")
ax5.set_xlabel("Horas relativas (0 = ahora)")
ax5.set_ylabel("PM2.5")
ax5.set_title("Pronóstico con cono de incertidumbre (próximas horas)")
ax5.legend()
ax5.grid(True, alpha=0.3)
st.pyplot(fig5)

st.markdown(
    """
Lista de pronósticos medios por hora (con IC 90%):

"""
)

for i in range(n_future):
    st.write(
        f"Hora +{i + 1}: {mean_prediction[i]:.2f} PM2.5 "
        f"(IC 90%: [{lower_bound[i]:.2f}, {upper_bound[i]:.2f}])"
    )
