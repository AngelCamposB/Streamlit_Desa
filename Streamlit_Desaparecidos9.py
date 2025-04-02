# -------------------------
# Aplicación de Monitoreo de Personas Desaparecidas
# -------------------------
import re
import os
import time
import signal
import threading
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from io import BytesIO
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# -------------------------
# Carga y limpieza de datos
# -------------------------
@st.cache_data
def load_data(uploaded_file=None, default_filepath=None):
    """
    Carga datos desde un archivo Excel, con prioridad al archivo subido.
    
    Args:
        uploaded_file: Archivo subido por el usuario
        default_filepath: Ruta del archivo por defecto
    
    Returns:
        DataFrame con los datos, o None si no se puede cargar
    """
    try:
        # Primero intenta cargar el archivo subido
        if uploaded_file is not None:
            return pd.read_excel(uploaded_file), uploaded_file.name
        
        # Si no hay archivo subido, intenta cargar el archivo por defecto
        if default_filepath is not None:
            return pd.read_excel(default_filepath), os.path.basename(default_filepath)
        
        # Si no hay ningún archivo
        return None, None
    
    except FileNotFoundError:
        st.warning(f"⚠️ No se encontró el archivo Excel en: {default_filepath}")
        return None, None
    except PermissionError:
        st.error(f"🚫 Error de permisos al intentar acceder al archivo: {default_filepath}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo Excel: {e}")
        return None, None

def convertir_a_fecha(valor):
            if pd.isna(valor) or valor == "": #verifica si el valor es nulo o vacio
                return np.nan #Retorna Not a Time, para que luego se pueda reemplazar con el valor de la columna CI_FECDEN.
            elif isinstance(valor, int):
                return pd.to_datetime(f'{valor}-01-01')
            elif isinstance(valor, str) and valor.isdigit() and len(valor) == 4:
                return pd.to_datetime(f'{valor}-01-01')
            else:
                try:
                    return pd.to_datetime(valor)
                except ValueError:
                    return pd.NaT
                
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la limpieza básica de los datos."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    if list(df.columns) == list(range(len(df.columns))):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    columnas_numericas = ["PD_EDAD"]
    columnas_fecha = ["CI_FECDEN", "DD_FECDESAP"]
    columnas_texto = ["PD_SEXO", "DD_MPIO", "DD_ESTADO", "CI_CARPEINV"]

    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in columnas_fecha:
        if col in df.columns:
            df[col] = df[col].apply(convertir_a_fecha)

    if "DD_FECDESAP" in df.columns and "CI_FECDEN" in df.columns:
         df['DD_FECDESAP'] = df['DD_FECDESAP'].fillna(df['CI_FECDEN'])

    if "PD_SEXO" in df.columns:
        mapeo_sexo = {
            'm': 'Hombre', 'masculino': 'Hombre', 'hombre': 'Hombre',
            'f': 'Mujer', 'femenino': 'Mujer', 'mujer': 'Mujer'
        }
        df["PD_SEXO"] = df["PD_SEXO"].str.strip().str.lower().map(mapeo_sexo).fillna(df["PD_SEXO"].str.capitalize())

    for col in columnas_texto:
        if col in df.columns and col != "PD_SEXO":
            df[col] = df[col].astype(str).str.strip().str.lower().str.capitalize()

    return df


# -------------------------
# Componentes de UI
# -------------------------
def setup_page_config():
    """Configura la página de Streamlit."""
    st.set_page_config(page_title="Monitoreo de Personas Desaparecidas", layout="wide")
    
    # Estilos personalizados
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #008CBA;
        color: black;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_date_filter(data):
    """Crea un filtro de fecha con botón de reinicio."""
    if "DD_FECDESAP" not in data.columns:
        return data.copy()
        
    fecha_min_original = data["DD_FECDESAP"].min().date()
    fecha_max_original = data["DD_FECDESAP"].max().date()

    # Inicializar el estado si no existe
    if 'rango_fechas' not in st.session_state:
        st.session_state['rango_fechas'] = [fecha_min_original, fecha_max_original]
    if 'date_input_key' not in st.session_state:
        st.session_state['date_input_key'] = 'main_filter'

    col1, col2 = st.sidebar.columns([2, 1])
    
    with col1:
        rango_fechas = st.date_input(
            ":green[Rango] de :violet[Fechas de Desapariciones] para Métricas",
            st.session_state['rango_fechas'],
            key=st.session_state['date_input_key']
        )
    
    with col2:
        st.write("")
        if st.button("Reset", key="reset_date"):
            st.session_state['rango_fechas'] = [fecha_min_original, fecha_max_original]
            st.session_state['date_input_key'] = f'main_filter_{datetime.now().strftime("%H%M%S")}'
            # Incrementar la clave para forzar que se recreen los widgets
            st.session_state.filter_key += 1
            st.rerun()

    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        start_date, end_date = rango_fechas
        filtered_data = data[(data["DD_FECDESAP"].dt.date >= start_date) & 
                             (data["DD_FECDESAP"].dt.date <= end_date)].copy()
    else:
        filtered_data = data.copy()
        
    return filtered_data


# -------------------------
# Funciones de análisis y visualización
# -------------------------
def display_metrics(df: pd.DataFrame):
    """Calcula y muestra las métricas clave."""
    fecha_min, fecha_max = df["DD_FECDESAP"].min().date(), df["DD_FECDESAP"].max().date()
    st.subheader(f"Metricas de {fecha_min} a {fecha_max}")

    total = len(df)
    localizados = df["PD_ESTATUSVIC"].value_counts().get("LOCALIZADA", 0)
    no_localizados = total - localizados
    promedio = df["PD_EDAD"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(":violet[Total Reportes]", total)
    col2.metric("Total :green[Localizados]", localizados, f"{localizados/total*100:.2f}%")
    col3.metric("Total :red[No Localizados]", no_localizados, f"{no_localizados/total*100:.2f}%")
    col4.metric(":orange[Edad Promedio]", f"{promedio:.1f} años")
    return total, localizados

def display_metrics_reaparecidos(df: pd.DataFrame):
    """Calcula y muestra las métricas clave relacionadas con las reapariciones, corrigiendo errores de digitación."""
    fecha_min, fecha_max = df["DD_FECDESAP"].min().date(), df["DD_FECDESAP"].max().date()
    st.subheader(f"Metricas de {fecha_min} a {fecha_max}")

    total = len(df)
    localizados = df["PD_ESTATUSVIC"].value_counts().get("LOCALIZADA", 0)

    # Calcular métricas de localizados si la columna existe
    localizados_con_vida = 0
    localizados_sin_vida = 0

    def normalizar_estatus(valor):
        if valor is None:
            return None
        
        # Convertir a minúsculas y eliminar espacios extra
        valor_norm = str(valor).lower().strip()
        
        # Mapeo de variaciones a valores estándar
        if 'con vida' in valor_norm: return 'CON VIDA'
        elif 'sin vida' in valor_norm: return 'SIN VIDA'
        return valor

    if "DL_LOCALSVCV" in df.columns and localizados > 0:
        df_loc = df[df["PD_ESTATUSVIC"] == "LOCALIZADA"]
        status_counts = df_loc["DL_LOCALSVCV"].value_counts().apply(normalizar_estatus)
        localizados_con_vida = status_counts.get("CON VIDA", 0)
        localizados_sin_vida = status_counts.get("SIN VIDA", 0)
        con_sin_vida = localizados_con_vida + localizados_sin_vida

    # Calcular el tiempo promedio de reaparición (en días), excluyendo outliers y errores de digitación
    if "DD_FECDESAP" in df.columns and "DL_FECLOC" in df.columns:
        df["DL_FECLOC"] = pd.to_datetime(df["DL_FECLOC"], errors='coerce')
        df["TIEMPO_REAPARICION"] = (df["DL_FECLOC"] - df["DD_FECDESAP"]).dt.days

        # Filtrar outliers (1 año o más) y errores de digitación (tiempo negativo)
        df_filtrado = df[(df["TIEMPO_REAPARICION"] < 365) & (df["TIEMPO_REAPARICION"] >= 0)]

        tiempo_promedio_reaparicion = df_filtrado["TIEMPO_REAPARICION"].mean() if not df_filtrado.empty else np.nan
    else:
        st.error("Error con las columnas DD_FECDESAP y/o DL_FECLOC")
        tiempo_promedio_reaparicion = np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(":blue[Total Localizados]", localizados, f"{localizados/total*100:.2f}%")
    col2.metric("Localizados :green[con Vida]", localizados_con_vida, f"{localizados_con_vida/localizados*100:.2f}%" if not np.isnan(localizados_con_vida) and localizados > 0 else "N/A")
    col3.metric("Localizados :red[sin Vida]", localizados_sin_vida, f"{localizados_sin_vida/localizados*100:.2f}%" if not np.isnan(localizados_sin_vida) and localizados > 0 else "N/A")
    col4.metric(":orange[Tiempo Promedio Localización (sin val. atípicos)]", f"{tiempo_promedio_reaparicion:.1f} días" if not np.isnan(tiempo_promedio_reaparicion) else "N/A", help="Sin valores atipicos: Sin casos mayores a 1 año de tiempo de localizacion")
    if localizados_con_vida + localizados_sin_vida != localizados:
        st.info(f"Hay :orange[{localizados - con_sin_vida}] casos en los que no se registró si se localizaron :green[CON VIDA] o :red[SIN VIDA]")    
    return total, localizados

def plot_outliers_reapariciones(df: pd.DataFrame):
    """Visualiza los casos de reaparición de 1 año o más (outliers) y los errores de digitación."""
    st.markdown("---")
    st.subheader("Casos de :red[Reaparición Atípicos] y :red[Errores de Digitación]")
    if "DD_FECDESAP" in df.columns and "DL_FECLOC" in df.columns:
        # Convertir fechas y calcular tiempo de reaparición
        df["DL_FECLOC"] = pd.to_datetime(df["DL_FECLOC"], errors='coerce')
        df["TIEMPO_REAPARICION"] = (df["DL_FECLOC"] - df["DD_FECDESAP"]).dt.days

        # Calcular tiempo en años
        df["TIEMPO_REAPARICION_ANOS"] = df["TIEMPO_REAPARICION"] / 365.25
        
        # Filtrar datos inválidos de una sola vez - valores de tiempo negativos o nulos
        df_validos = df.dropna(subset=["TIEMPO_REAPARICION", "DD_FECDESAP"])
        
        if df_validos.empty:
            st.warning("No hay datos de reaparición disponibles. Revise Etiquetas de Filtro")
            return
        
        # Extraer año y mes para operaciones posteriores
        df_validos["YEAR"] = df_validos["DD_FECDESAP"].dt.year
        df_validos["MONTH"] = df_validos["DD_FECDESAP"].dt.month
        df_validos["PERIODO_DESAPARICION"] = df_validos["DD_FECDESAP"].dt.to_period('M')
        
        # Obtener solo los años y meses que existen en los datos
        months_years = df_validos.groupby(["YEAR", "MONTH"]).size().reset_index()
        
        # Mapeo español para nombres de meses
        month_names = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
            7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        
        # Mapeo inverso para convertir nombres de meses de español a números
        month_numbers = {v: k for k, v in month_names.items()}
        
        # Crear opciones para el slider solo con los periodos existentes
        combined = []
        for _, row in months_years.sort_values(by=["YEAR", "MONTH"]).iterrows():
            combined.append(f"{month_names[row['MONTH']]} {row['YEAR']}")
        
        if not combined:
            st.warning("No hay periodos válidos disponibles.")
            return
        
        # Slider para seleccionar rango
        selected_range = st.select_slider("Rango de Meses y Años", combined, value=(combined[0], combined[-1]))
        
        # Procesar selección
        start_month_name, start_year = selected_range[0].split()
        end_month_name, end_year = selected_range[1].split()
        
        # Convertir nombres de meses a números
        start_month_num = month_numbers[start_month_name]
        end_month_num = month_numbers[end_month_name]
        
        # Convertir a fechas
        start_date = pd.Timestamp(year=int(start_year), month=start_month_num, day=1)
        if int(end_month_num) == 12:
            end_date = pd.Timestamp(year=int(end_year), month=end_month_num, day=31)
        else:
            end_date = pd.Timestamp(year=int(end_year), month=int(end_month_num) + 1, day=1) - pd.Timedelta(days=1)
        
        # Filtrar por rango de fechas seleccionado
        filtro = df_validos[(df_validos["DD_FECDESAP"] >= start_date) & (df_validos["DD_FECDESAP"] <= end_date)]

        # Outliers (1 año o más)
        df_outliers = filtro[filtro["TIEMPO_REAPARICION_ANOS"] >= 1]

        # Errores de digitación (tiempo negativo)
        df_errores = filtro[filtro["TIEMPO_REAPARICION"] < 0]

        if not df_outliers.empty:
            st.write("#### Casos de Reaparición de 1 Año o Más :red[(Atípicos)]")
            
            # Calcular el máximo valor para el eje x
            max_value = df_outliers["TIEMPO_REAPARICION_ANOS"].max()
            # Asegurar que el valor máximo sea al menos 40
            max_x = max(40, max_value)
            # Añadir 5% adicional para que se muestre el número 40
            max_x_display = max_x * 1.05
            
            # Crear el histograma con ajustes
            fig = px.histogram(
                df_outliers,
                x="TIEMPO_REAPARICION_ANOS",
                nbins=50,
                labels={
                    "TIEMPO_REAPARICION_ANOS": "Tiempo de Reaparición (años)",
                    "count": "Número de Casos"
                },
                title=f"Distribución del Tiempo de Reaparición (Outliers) - {len(df_outliers)} casos",
            )
            
            # Ajuste del eje x para que empiece en 1 y termine un poco después de 40
            fig.update_layout(
                xaxis=dict(
                    range=[1, max_x_display],  # Establece el rango mínimo en 1
                    dtick=5,                   # Establece las marcas principales cada 5 años
                ),
                yaxis=dict(
                    title="Número de Casos"     # Cambia la etiqueta del eje y
                )
            )
            
            # Ajuste de las barras para que estén correctamente alineadas
            fig.update_traces(
                xbins=dict(
                    start=1,          # Inicia las barras exactamente en 1
                    end=max_x_display,  # Establece el fin para incluir el valor 40
                    size=1            # Tamaño de cada barra (1 año)
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Create two columns for the tables
            col1, col2 = st.columns(2)

            with col1:
                st.write("")
                st.write("")
                st.write(f"#### Detalles de :red[Valores Atípicos] - Total: {len(df_outliers)}")
                df_outliers_display = df_outliers[["DD_FECDESAP", "DL_FECLOC", "TIEMPO_REAPARICION"]]
                df_outliers_display.index += 2
                st.dataframe(df_outliers_display)

            with col2:
                if not df_errores.empty:
                    st.write(f"#### :red[Errores de Digitación] (Tiempo de Desaparición Negativo) - Total: {len(df_errores)}")
                    df_errores_display = df_errores[["DD_FECDESAP", "DL_FECLOC", "TIEMPO_REAPARICION"]]
                    df_errores_display.index += 2
                    st.dataframe(df_errores_display)
                else:
                    st.info("No se encontraron errores de digitación en las fechas en el rango seleccionado.")

        else:
            st.warning("No se encontraron casos de reaparición de 1 año o más en el rango seleccionado.")

            if not df_errores.empty:
                st.write(f"#### Errores de Digitación (Tiempo de Desaparición Negativo) - Total: {len(df_errores)}")
                df_errores_display = df_errores[["DD_FECDESAP", "DL_FECLOC", "TIEMPO_REAPARICION"]]
                df_errores_display.index += 2
                st.dataframe(df_errores_display)
            else:
                st.info("No se encontraron errores de digitación en las fechas en el rango seleccionado.")
    else:
        st.warning("Las columnas 'DD_FECDESAP' y/o 'DL_FECLOC' no se encontraron.")


def plot_desapariciones_por_año(df: pd.DataFrame):
    """
    Grafica la tendencia de desapariciones por año usando DD_FECDESAP.
    Muestra un slider para ajustar el intervalo visualizado.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de desapariciones
    
    Returns:
        None
    """
    if "DD_FECDESAP" not in df.columns or "PD_ESTATUSVIC" not in df.columns:
        st.warning("Columnas necesarias no encontradas.")
        return
    
    # Procesamiento de datos
    df["AÑO_DESAPARICION"] = df["DD_FECDESAP"].dt.year
    grupo = df.groupby(["AÑO_DESAPARICION", "PD_ESTATUSVIC"]).size().unstack(fill_value=0)
    
    # Aseguramos que ambas categorías estén presentes
    if "LOCALIZADA" not in grupo.columns:
        grupo["LOCALIZADA"] = 0
    if "EN INVESTIGACION" not in grupo.columns:
        grupo["EN INVESTIGACION"] = 0
    
    grupo["TOTAL"] = grupo.sum(axis=1)
    grupo.index = grupo.index.astype(int)
    grupo.index.name = "AÑO_DESAPARICION"
    
    # Aseguramos que todos los años estén representados
    all_years = range(grupo.index.min(), grupo.index.max() + 1)
    grupo = grupo.reindex(all_years, fill_value=0).reset_index()
    
    # Visualización
    st.subheader(" Registro (Histórico) de Desapariciones por Año")
    min_year, max_year = int(df["AÑO_DESAPARICION"].min()), int(df["AÑO_DESAPARICION"].max())
    selected_range = st.slider("Intervalo de años (Desapariciones)", min_year, max_year, (min_year, max_year))
    filtro = grupo[(grupo["AÑO_DESAPARICION"] >= selected_range[0]) & (grupo["AÑO_DESAPARICION"] <= selected_range[1])]

    fig = px.line(filtro, x="AÑO_DESAPARICION", y=["TOTAL", "LOCALIZADA", "EN INVESTIGACION"],
                  labels={"AÑO_DESAPARICION": "Año", "value": "Cantidad"},
                  markers=True,
                  color_discrete_map={"TOTAL": "blue", "LOCALIZADA": "green", "EN INVESTIGACION": "orange"})
    fig.update_layout(legend_title_text='')
    st.plotly_chart(fig, use_container_width=True)

def procesar_datos_ARIMA(df: pd.DataFrame):
    """
    Procesa los datos de registros por año con opción de seleccionar 
    entre desapariciones o denuncias.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de registros
        
    Returns:
        tuple: (training_series, año_actual_valor)
            - training_series: Serie completa de registros por año para entrenamiento (2010 en adelante)
            - año_actual_valor: Valor de registros para el año actual
    """
    # Selector de tipo de registro
    tipo_registro = st.radio("Seleccione sobre qué quiere hacer la predicción", ["Desapariciones", "Denuncias"], horizontal=True)
    
    # Seleccionar la columna de fecha basada en la selección del usuario
    columna_fecha = "DD_FECDESAP" if tipo_registro == "Desapariciones" else "CI_FECDEN"
    
    # Validar la existencia de la columna de fecha y el estatus de víctima
    if columna_fecha not in df.columns or "PD_ESTATUSVIC" not in df.columns:
        st.warning(f"Columna {columna_fecha} o PD_ESTATUSVIC no encontrada.")
        return pd.Series(), 0
    
    # Procesamiento de datos
    df["AÑO_REGISTRO"] = df[columna_fecha].dt.year
    grupo = df.groupby(["AÑO_REGISTRO", "PD_ESTATUSVIC"]).size().unstack(fill_value=0)
    
    # Aseguramos que ambas categorías estén presentes
    if "LOCALIZADA" not in grupo.columns:
        grupo["LOCALIZADA"] = 0
    if "EN INVESTIGACION" not in grupo.columns:
        grupo["EN INVESTIGACION"] = 0
    
    grupo["TOTAL"] = grupo.sum(axis=1)
    grupo.index = grupo.index.astype(int)
    grupo.index.name = "AÑO_REGISTRO"
    
    # Aseguramos que todos los años estén representados
    all_years = range(grupo.index.min(), grupo.index.max() + 1)
    grupo = grupo.reindex(all_years, fill_value=0).reset_index()
    
    # Obtener el año actual
    año_actual = datetime.now().year
    
    # Preparar datos para entrenamiento (desde 2010)
    all_years_training = range(2010, año_actual)
    training_data = grupo.set_index("AÑO_REGISTRO").reindex(all_years_training, fill_value=0)
    training_series = training_data["TOTAL"]
    
    # Valor del año actual
    año_actual_valor = grupo[grupo["AÑO_REGISTRO"] == año_actual]["TOTAL"].iloc[0] if not grupo[grupo["AÑO_REGISTRO"] == año_actual].empty else 0
    
    return training_series, año_actual_valor
    
@st.cache_resource
def entrenar_modelo_arima(data_training):
    """Entrena el modelo ARIMA con los datos proporcionados."""
    return ARIMA(data_training.iloc[:,1], order=(1, 1, 1)).fit()

@st.cache_data
def generar_prediccion_arima(_modelo, data_traning, future_years=6, conf_level=0.95):
    """Genera la predicción con intervalos de confianza."""
    forecast = _modelo.get_forecast(steps=future_years)
    years = range(data_traning.iloc[:,0].max() + 1, data_traning.iloc[:,0].max() + future_years + 1)
    conf_int = forecast.conf_int(alpha=1 - conf_level)
    forecast = forecast.predicted_mean

    # Crear el DataFrame con las predicciones y los intervalos de confianza
    df_predicciones = pd.DataFrame({
        'year': years,
        'prediccion': forecast,
        'lower_bound': conf_int.iloc[:, 0],
        'upper_bound': conf_int.iloc[:, 1]
    })

    # Aplicar np.maximum directamente a la columna 'prediccion'
    df_predicciones.loc[df_predicciones["lower_bound"] <= 0, "lower_bound"] = 0

    return df_predicciones

def mostrar_prediccion(predictions, data_traning, confidence_level, valor_real_año_actual):
    """Muestra las predicciones y el gráfico."""
    st.subheader("Predicciones de ARIMA")
    st.dataframe(predictions.round(1))

    ultimo_año = data_traning.iloc[:, 0].max()
    año_actual = ultimo_año + 1

    pred_futuro = predictions[predictions['year'] == año_actual].iloc[0]
    valor_predicho, limite_inferior, limite_superior = pred_futuro['prediccion'], pred_futuro['lower_bound'], pred_futuro['upper_bound']
    dentro_intervalo = limite_inferior <= valor_real_año_actual <= limite_superior

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_traning.iloc[:, 0], y=data_traning.iloc[:, 1], mode='lines+markers', name='Datos Históricos'))
    fig.add_trace(go.Scatter(x=[año_actual], y=[valor_real_año_actual], mode='markers', name=f'Valor Real {año_actual}'))
    fig.add_trace(go.Scatter(x=predictions['year'], y=predictions['prediccion'], mode='lines+markers', name='Predicción ARIMA'))

    conf_x = list(predictions['year'])
    conf_y_upper = list(predictions['upper_bound'])
    conf_y_lower = list(predictions['lower_bound'])

    # Invertir y extender las listas de confianza
    conf_x.extend(list(reversed(conf_x)))
    conf_y_upper.extend(list(reversed(conf_y_lower)))

    fig.add_trace(go.Scatter(x=conf_x, y=conf_y_upper, fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,0,0,0)'), name=f'Intervalo de Confianza ({confidence_level}%)'))
    fig.update_layout(title='Predicción de Personas Desaparecidas con ARIMA', xaxis_title='Año', yaxis_title='Número de Personas Desaparecidas', xaxis=dict(tickmode='linear', tick0=data_traning.iloc[:,0].min(), dtick=1), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    rango = "del rango de predicción"
    col1.metric(f"Predicción {año_actual}", f"{valor_predicho:.0f}", f"{valor_predicho - valor_real_año_actual:.0f} de diferencia")
    col2.metric("Intervalo de Confianza", f"[{limite_inferior:.0f} - {limite_superior:.0f}]")
    col3.metric(f"Valor Real Actual {año_actual}", f"{valor_real_año_actual}", f"Dentro {rango}" if dentro_intervalo else f"Fuera {rango}", "normal" if dentro_intervalo else "inverse")

    st.header("Detalles del Modelo ARIMA")
    st.write("Este modelo utiliza ARIMA (Autoregressive Integrated Moving Average) para predecir el número de personas desaparecidas en años futuros. La configuración del modelo es ARIMA(1,1,1).")
    # Explicación de los intervalos
    st.info(f"""
    **Sobre los intervalos de confianza:**
    - El intervalo mostrado tiene una confianza del {confidence_level}%.
    - Esto significa que hay un {confidence_level}% de probabilidad de que el valor real esté dentro de este rango.
    - A medida que aumenta el nivel de confianza, el intervalo de confianza se vuelve más amplio, lo que significa que la predicción es menos precisa.
    """)

    # Advertencia sobre la precisión
    st.warning("""
    **Nota importante:** Las predicciones son estimaciones basadas en patrones históricos. Factores externos no considerados en el modelo pueden afectar significativamente los resultados futuros.
               Para mayor presición se requieren datos de Factores socioeconómicos, de seguridad pública, demográficos, etc.
    """)
###############

def plot_denuncias_por_año(df: pd.DataFrame):
    """Grafica la tendencia de denuncias por año (CI_FECDEN) desde 2001."""
    if "CI_FECDEN" not in df.columns or "PD_ESTATUSVIC" not in df.columns:
        st.warning("Columnas CI_FECDEN y/o PD_ESTATUSVIC no encontradas.")
        return
    
    # Filtrar primero antes de cualquier operación
    mask = (df["CI_FECDEN"].dt.year >= 2001)
    if not mask.any():
        st.warning("No hay datos de denuncias desde 2001.")
        return
    
    # Usar solo las columnas necesarias sin crear copia completa
    # view() no funciona como se esperaría en pandas - no es más eficiente que loc
    df_temp = df.loc[mask, ["CI_FECDEN", "PD_ESTATUSVIC"]]
    df_temp["AÑO_DENUNCIAS"] = df_temp["CI_FECDEN"].dt.year
    
    # Crear pivot_table directamente con las columnas específicas que necesitamos
    # y usar reindex para asegurar columnas necesarias sin bucles
    conteo = pd.pivot_table(
        df_temp,
        index="AÑO_DENUNCIAS",
        columns="PD_ESTATUSVIC",
        aggfunc="size",
        fill_value=0
    ).reset_index()
    
    # Asegurar que existan las columnas necesarias de manera más eficiente
    cols = ["LOCALIZADA", "EN INVESTIGACION"]
    missing_cols = [col for col in cols if col not in conteo.columns]
    if missing_cols:
        for col in missing_cols:
            conteo[col] = 0
    
    # Calcular el total solo sumando las columnas relevantes
    conteo["TOTAL"] = conteo[["LOCALIZADA", "EN INVESTIGACION"]].sum(axis=1)
    
    # UI y visualización
    st.subheader("📅 Registro de Denuncias por Año")
    min_year, max_year = int(conteo["AÑO_DENUNCIAS"].min()), int(conteo["AÑO_DENUNCIAS"].max())
    selected_range = st.slider("Intervalo de años (Denuncias) - Se toman los años en que se hicieron las denuncias",
                               min_year, max_year, (min_year, max_year), key="slider_denuncias", help="Se toman los años en que se hicieron las denuncias")
    
    # Filtrar usando índices booleanos para mejor rendimiento
    mask_filtro = (conteo["AÑO_DENUNCIAS"] >= selected_range[0]) & (conteo["AÑO_DENUNCIAS"] <= selected_range[1])
    if mask_filtro.any():
        filtro = conteo.loc[mask_filtro]
        fig = px.line(filtro, x="AÑO_DENUNCIAS", y=["TOTAL", "LOCALIZADA", "EN INVESTIGACION"],
                      labels={"AÑO_DENUNCIAS": "Año", "value": "Cantidad"},
                      markers=True,
                      color_discrete_map={"TOTAL": "purple", "LOCALIZADA": "green", "EN INVESTIGACION": "orange"})
        fig.update_layout(legend_title_text='')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos en el rango seleccionado.")


def plot_expedientes_por_año(df: pd.DataFrame):
    """Grafica la tendencia de expedientes de desapariciones por año (CI_CARPEINV) usando CI_FECDEN."""
    if "CI_FECDEN" not in df.columns or "CI_CARPEINV" not in df.columns:
        st.warning("Columnas CI_FECDEN y/o CI_CARPEINV no encontradas.")
        return
    
    # Filtrar primero solo con las columnas necesarias
    mask = df[["CI_FECDEN", "CI_CARPEINV"]].notna().all(axis=1)
    if not mask.any():
        st.warning("No hay datos de expedientes disponibles.")
        return
    
    # Seleccionar solo las columnas necesarias sin crear una copia completa
    df_temp = df.loc[mask, ["CI_FECDEN", "CI_CARPEINV"]]
    # Extraer el año directamente sin crear columna adicional
    años = df_temp["CI_FECDEN"].dt.year
    
    # Agrupar y contar de manera más eficiente
    conteo = df_temp.groupby(años)["CI_CARPEINV"].nunique()
    conteo = conteo.reset_index(name="NUM_EXPEDIENTES")
    conteo.columns = ["AÑO_DENUNCIAS", "NUM_EXPEDIENTES"]  # Renombrar después del reset
    
    # UI y visualización
    st.subheader("Registro de Expedientes de Desapariciones por Año")
    min_year, max_year = int(conteo["AÑO_DENUNCIAS"].min()), int(conteo["AÑO_DENUNCIAS"].max())
    selected_range = st.slider("Intervalo de años (Expedientes) - Se toman los años en que se hicieron las denuncias",
                               min_year, max_year, (min_year, max_year), key="slider_expedientes")
    
    # Filtrar usando índices booleanos para mejor rendimiento
    mask_filtro = (conteo["AÑO_DENUNCIAS"] >= selected_range[0]) & (conteo["AÑO_DENUNCIAS"] <= selected_range[1])
    if mask_filtro.any():
        filtro = conteo.loc[mask_filtro]
        fig = px.line(filtro, x="AÑO_DENUNCIAS", y="NUM_EXPEDIENTES",
                      labels={"AÑO_DENUNCIAS": "Año", "NUM_EXPEDIENTES": "Número de Expedientes"},
                      markers=True,
                      title="Número de Expedientes de Desaparición por Año")
        fig.update_layout(legend_title_text='')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos en el rango de años seleccionado.")

def plot_sexo_distribution(df: pd.DataFrame):
    """Grafica la distribución por sexo (usando CI_FECDEN para filtrar años)."""
    st.subheader("👥 Distribución por Sexo")
    if "PD_SEXO" in df.columns and "CI_FECDEN" in df.columns:
        df["AÑO_DENUNCIAS"] = df["CI_FECDEN"].dt.year
        min_year = int(df["AÑO_DENUNCIAS"].min())
        max_year = int(df["AÑO_DENUNCIAS"].max())
        selected_range = st.slider("Intervalo de años (Sexo) - Se toman los años en que se hicieron las denuncias", min_year, max_year, (min_year, max_year), key="slider_sexo", help="Se toman los años de las denuncias")
        filtro = df[(df["AÑO_DENUNCIAS"] >= selected_range[0]) & (df["AÑO_DENUNCIAS"] <= selected_range[1])]
        counts = filtro["PD_SEXO"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     title="Distribución de Denuncias por Sexo")
        fig.update_traces(textinfo="percent+label", insidetextorientation="radial")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columnas necesarias no encontradas.")

@st.fragment
def plot_clasedad_distribution(df: pd.DataFrame):
    """Grafica la distribución por clasificación etaria (usando CI_FECDEN)."""
    st.subheader("Distribución por Clasificación de Edad")
    if "PD_CLASEDAD" in df.columns and "CI_FECDEN" in df.columns:
        df["AÑO_DENUNCIAS"] = df["CI_FECDEN"].dt.year
        min_year = int(df["AÑO_DENUNCIAS"].min())
        max_year = int(df["AÑO_DENUNCIAS"].max())
        selected_range = st.slider("Intervalo de años (Clasificación de Edad) - Se toman los años de las denuncias", min_year, max_year, (min_year, max_year), help="Se toman los años de las denuncias",key="slider_clasedad")
        filtro = df[(df["AÑO_DENUNCIAS"] >= selected_range[0]) & (df["AÑO_DENUNCIAS"] <= selected_range[1])]
        counts = filtro["PD_CLASEDAD"].value_counts()
        total = counts.sum()
        pct = (counts / total) * 100
        fig = px.bar(x=counts.index, y=counts.values,
                     labels={"x": "Clasificación de Edad", "y": "Cantidad"},
                     title="Distribución de Denuncias por Clasificación de Edad")
        for i, count in enumerate(counts):
            fig.add_annotation(x=counts.index[i], y=count,
                               text=f"{count} ({pct.iloc[i]:.1f}%)", yanchor="bottom")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columnas necesarias no encontradas.")

@st.fragment
def plot_age_distribution(df: pd.DataFrame):
    """Grafica la distribución de edades con opciones para agrupar datos y muestra tabla de detalles."""
    st.subheader("📊 Distribución de Edades")
    
    # Verificar si existe la columna de edad
    if "PD_EDAD" not in df.columns:
        st.warning("Columna de edad no encontrada.")
        return
    
    # Preparar datos y aplicar filtros
    year_filter_applied = False

    # Aplicar filtros usando máscaras booleanas en lugar de crear copias
    if "CI_FECDEN" in df.columns:
        años_denuncias = df["CI_FECDEN"].dt.year
        min_year = int(años_denuncias.min())
        max_year = int(años_denuncias.max())
        selected_range = st.slider("Intervalo de años - Se toman los años de las denuncias", min_year, max_year, (min_year, max_year),
                                   help="Se toman los años de las denuncias", key="slider_edades")
        
        # Crear máscara para filtrar sin copiar el dataframe
        mask_años = (años_denuncias >= selected_range[0]) & (años_denuncias <= selected_range[1])
        year_filter_applied = True
    else:
        # Si no hay columna de fechas, usamos todos los datos
        mask_años = pd.Series(True, index=df.index)

    # Datos de edad para análisis
    # Extraer edades aplicando la máscara directamente
    edades = df.loc[mask_años, "PD_EDAD"].dropna()
    
    if len(edades) == 0:
        st.warning("No hay datos de edad disponibles para el filtro seleccionado.")
        return
        
    min_edad = int(edades.min())
    max_edad = int(edades.max())
    
    # Configuración de controles y estadísticas en columnas
    col1, col2 = st.columns([1, 1])
    
    # Columna 1: Controles de visualización
    with col1:
        # Opciones de visualización
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            intervalo_edad = st.number_input("Tamaño del intervalo", 
                                        min_value=1, max_value=10, value=5, 
                                        help="Define el rango de años para cada grupo")
        with subcol2:
            vista = st.radio("Tipo de visualización", 
                        ["Barras", "Área"], 
                        horizontal=True)
        
        # Crear los intervalos de edad una sola vez
        max_bin = max_edad + (intervalo_edad - max_edad % intervalo_edad) if max_edad % intervalo_edad != 0 else max_edad
        bins = list(range(min_edad, max_bin + intervalo_edad, intervalo_edad))
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        
        # Calcular distribución de edades por intervalo
        edad_intervalo = pd.cut(edades, bins=bins, right=False, labels=labels)
        counts = edad_intervalo.value_counts().sort_index()
        total = len(edades)
        
        # Mostrar el rango más frecuente
        rango_max = counts.idxmax()
        count_max = counts.max()
        pct_max = (count_max/total*100)
        st.info(f"💡 El rango de edad más común es **{rango_max}** con **{count_max}** casos ({pct_max:.1f}% del total)")
        
        if year_filter_applied or len(edades) < len(df):
            st.warning("La ausencia de algunos registros de edad hace que el total de casos para esta grafica no coincida con el total general de reportes registrados.")
    
    # Columna 2: Estadísticas descriptivas
    with col2:
        st.write("**Estadísticas de edad**")
        
        # Primera fila de métricas
        metrics_row1 = st.columns(3)
        metrics_row1[0].metric("Edad promedio", f"{edades.mean():.1f} años")
        metrics_row1[1].metric("Mediana", f"{edades.median():.1f} años")
        metrics_row1[2].metric("Moda", f"{edades.mode().iloc[0]:.0f} años")
        
        # Segunda fila de métricas
        metrics_row2 = st.columns(3)
        metrics_row2[0].metric("Mínimo", f"{min_edad:.0f} años")
        metrics_row2[1].metric("Máximo", f"{max_edad:.0f} años")
        metrics_row2[2].metric("Total casos", f"{len(edades):,}")
    
    # Calcular porcentajes para anotaciones
    pct = (counts / total) * 100
    
    # Configuración base del gráfico
    fig_config = {
        "x": counts.index,
        "y": counts.values,
        "labels": {"x": "Rango de Edad", "y": "Cantidad"},
        "title": "Distribución de Edades",
        "color_discrete_sequence": ["#3366CC"]
    }
    
    # Crear el gráfico según el tipo seleccionado
    fig = px.bar(**fig_config) if vista == "Barras" else px.area(**fig_config)
    
    # Añadir línea de tendencia si es gráfico de barras
    if vista == "Barras":
        fig.add_scatter(x=counts.index, y=counts.values, mode='lines', 
                       line=dict(color='#FF6600', width=2), name='Tendencia')
    
    # Añadir anotaciones con valores y porcentajes
    for i, (idx, value) in enumerate(zip(counts.index, counts.values)):
        fig.add_annotation(
            x=idx, y=value,
            text=f"{value} ({pct.iloc[i]:.1f}%)",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255, 255, 255, 0.7)"
        )
    
    # Estilo del gráfico
    fig.update_layout(
        xaxis_title="Rango de Edad",
        yaxis_title="Cantidad",
        hoverlabel=dict(bgcolor="blue"),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#8c8c8c",
        paper_bgcolor="#a9a9a9",
        font=dict(color="black", size=12)
    )
    
    # Configuración de ejes y cuadrícula
    for axis_update in [fig.update_xaxes, fig.update_yaxes]:
        axis_update(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='#454546',
            title_font=dict(color="black", size=12),
            tickfont=dict(color="black", size=11)
        )
    
    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Preparar y mostrar tabla de detalles
    st.markdown("#### Detalles de Registros")

    # Filtrar el dataframe original con la máscara de años
    df_filtrado = df.loc[mask_años]

    # Seleccionar las columnas requeridas
    columnas_tabla = ["index", "PD_EDAD", "PD_CLASEDAD", "PD_SEXO", "PD_ESTATUSVIC"]

    # Verificar si todas las columnas existen
    columnas_existentes = [col for col in columnas_tabla if col in df_filtrado.columns]

    if len(columnas_existentes) > 1:
        # Añadir el índice original como columna
        df_tabla = df_filtrado[columnas_existentes].copy()
        df_tabla.index = df_tabla.index + 2  # Cambiar el índice para que comience en 2
        
        # Renombrar columnas para mayor claridad
        nombre_columnas = {
            "index": "Índice Original",
            "PD_EDAD": "Edad",
            "PD_CLASEDAD": "Clasificación Edad",
            "PD_SEXO": "Sexo",
            "PD_ESTATUSVIC": "Estatus"
        }
        df_tabla.rename(columns=nombre_columnas, inplace=True)

        # Crear un expander para las tablas
        with st.expander("Detalles Completos"):
            # Dividir el expander en dos columnas
            col1, col2 = st.columns(2)
            
            # Columna 1: Tabla de Detalles
            with col1:
                personas_sin_edad = df_tabla["Edad"].isna().sum()
                st.write("### Tabla de Registros")
                st.write(f"Número de registros sin edad: {personas_sin_edad}")
                st.dataframe(df_tabla, use_container_width=True)
            
            # Columna 2: Tabla de Errores
            with col2:
                st.write("### :red[Errores] de Registro de Edad")
                # Validación de clasificación de edad
                df_errores = validar_clasificacion_edad(df_tabla)

                # Mostrar tabla de errores
                numero_errores = len(df_errores)    # Contar número de errores
                st.write(f"Total de registros con error: {numero_errores}")
                st.dataframe(df_errores, use_container_width=True)

    else:
        st.warning("No se encontraron todas las columnas requeridas para la tabla de detalles.")

###################################
# ---- ERRORES y VALIDACIONES ----#
###################################
def validar_clasificacion_edad(df):
    # Crear una copia del DataFrame para no modificar el original
    df_validado = df.copy()
    
    # Identificar errores de clasificación de edad
    df_errores = df_validado[
        (df_validado['Edad'] < 0) |  # Edad negativa
        ((df_validado['Edad'] >= 0) & (df_validado['Edad'] <= 11) & (df_validado['Clasificación Edad'].str.lower() != 'niño')) |  # Niño
        ((df_validado['Edad'] >= 12) & (df_validado['Edad'] <= 17) & (df_validado['Clasificación Edad'].str.lower() != 'adolescente')) |  # Adolescente
        ((df_validado['Edad'] >= 18) & (df_validado['Edad'] <= 59) & (df_validado['Clasificación Edad'].str.lower() != 'adulto')) |  # Adulto
        ((df_validado['Edad'] >= 60) & (df_validado['Clasificación Edad'].str.lower() != 'adulto mayor'))  # Adulto Mayor
    ]
    
    return df_errores


def validar_fechas_desaparicion(df):
    """
    Valida la cronología de fechas relacionadas con una desaparición.
    
    Parámetros:
    df (pandas.DataFrame): DataFrame con columnas de fechas a validar
    
    Retorna:
    tuple: (DataFrame de errores cronológicos, DataFrame de fechas incompletas)
    """
    # Seleccionar solo las columnas necesarias
    columnas_necesarias = [
        "PD_NOMBRECOMPLETOVICFORMULA", 'PD_ESTATUSVIC', 'PD_SEXO', 'PD_EDAD', 
        'DD_FECDESAP', 'DD_FECPERCA', 'CI_FECDEN', 'DL_FECLOC'
    ]
    
    # Crear una copia del DataFrame con solo las columnas necesarias
    # Incluir el índice original como columna
    df_validacion = df[columnas_necesarias].copy()
    #df_validacion['indice'] = df.index
    df_validacion.index = df_validacion.index + 2  # Ajustar índices sumando 2
    
    # Función para convertir fechas con manejo de errores
    def parse_fecha(fecha):
        try:
            # Intentar convertir a datetime, ignorando errores de formato
            return pd.to_datetime(fecha, errors='coerce')
        except:
            return pd.NaT
    
    # Convertir todas las columnas de fecha
    columnas_fecha = ['DD_FECDESAP', 'DD_FECPERCA', 'CI_FECDEN', 'DL_FECLOC']
    for col in columnas_fecha:
        df_validacion[col] = df_validacion[col].apply(parse_fecha)
    
    # Función para validar la secuencia cronológica
    def validar_cronologia(row):
        # Verificar que todas las fechas sean válidas
        fechas = [row['DD_FECDESAP'], row['DD_FECPERCA'], row['CI_FECDEN'], row['DL_FECLOC']]
        
        # Verificar si hay fechas no válidas
        if any(pd.isnull(fecha) for fecha in fechas if fecha is not pd.NaT):
            return True
        
        # Verificar la secuencia cronológica excluyendo fechas NaT
        fechas_validas = [f for f in fechas if not pd.isnull(f)]
        
        # Si hay menos de 2 fechas válidas, no podemos hacer la comparación
        if len(fechas_validas) < 2:
            return False
        
        # Verificar la secuencia cronológica
        # DD_FECDESAP <= DD_FECPERCA <= CI_FECDEN <= DL_FECLOC
        for i in range(len(fechas_validas) - 1):
            if fechas_validas[i] > fechas_validas[i+1]:
                return True
        
        return False
    
    # Aplicar validación y crear DataFrame de errores
    df_errores = df_validacion[df_validacion.apply(validar_cronologia, axis=1)].copy()
    
    # Separar filas con DL_FECLOC en 1970-01-01 
    # Usar condición que verifica año, mes y día
    df_fechas_incompletas = df_errores[
        (df_errores['DL_FECLOC'].dt.year == 1970) & 
        (df_errores['DL_FECLOC'].dt.month == 1) & 
        (df_errores['DL_FECLOC'].dt.day == 1)
    ].copy()
    
    # Quitar filas de fechas incompletas del DataFrame de errores
    df_errores = df_errores[
        ~((df_errores['DL_FECLOC'].dt.year == 1970) & 
          (df_errores['DL_FECLOC'].dt.month == 1) & 
          (df_errores['DL_FECLOC'].dt.day == 1))
    ]
    
    # Reordenar columnas para que 'indice' sea la primera
    columnas = [
        "PD_NOMBRECOMPLETOVICFORMULA", 'PD_ESTATUSVIC', 'PD_SEXO', 'PD_EDAD', 
        'DD_FECDESAP', 'DD_FECPERCA', 'CI_FECDEN', 'DL_FECLOC'
    ]
    
    df_errores = df_errores[columnas]
    df_fechas_incompletas = df_fechas_incompletas[columnas]
    
    # Mostrar con Streamlit
    st.write(f"### :red[Errores] Cronológicos: {len(df_errores)}")
    if not df_errores.empty:
        st.dataframe(df_errores)
    else:
        st.write("No se encontraron errores cronológicos.")
    
    st.write(f"### Solo Años - :red[Fechas incompletas]: {len(df_fechas_incompletas)}")
    if not df_fechas_incompletas.empty:
        st.dataframe(df_fechas_incompletas)
    else:
        st.write("No se encontraron fechas incompletas.")
    
    return df_errores, df_fechas_incompletas


@st.fragment
def plot_municipio_distribution(df: pd.DataFrame):
    """Grafica la distribución por municipio (usando CI_FECDEN)."""
    st.subheader("🏙️ Distribución por Municipio")
    if "DD_MPIO" in df.columns and "CI_FECDEN" in df.columns and "DD_ESTADO" in df.columns:
        df["AÑO_DENUNCIAS"] = df["CI_FECDEN"].dt.year
        min_year = int(df["AÑO_DENUNCIAS"].min())
        max_year = int(df["AÑO_DENUNCIAS"].max())
        selected_range = st.slider("Intervalo de años (Municipio) - Se toman los años de las denuncias", min_year, max_year, (min_year, max_year), key="slider_municipio", help="Se toman los años de las denuncias")
        filtro = df[(df["AÑO_DENUNCIAS"] >= selected_range[0]) & (df["AÑO_DENUNCIAS"] <= selected_range[1])]
        
        # Gráfica de los 10 municipios con más denuncias
        counts = filtro["DD_MPIO"].value_counts().nlargest(10)
        total = filtro["DD_MPIO"].value_counts().sum()
        pct = (filtro["DD_MPIO"].value_counts() / total) * 100
        fig = px.bar(x=counts.index, y=counts.values,
                     labels={"x": "Municipio", "y": "Cantidad"},
                     title="Top 10 Municipios con más Denuncias")
        for i, count in enumerate(counts):
            fig.add_annotation(x=counts.index[i], y=count,
                               text=f"{count} ({pct.iloc[i]:.1f}%)", yanchor="bottom")
        st.plotly_chart(fig, use_container_width=True)

        # Crear tabla completa sin duplicados
        # Agrupar por municipio y estado y contar las apariciones
        grouped_data = filtro.groupby(["DD_MPIO", "DD_ESTADO"]).size().reset_index(name="Cantidad")
        # Ordenar por cantidad descendente
        grouped_data = grouped_data.sort_values("Cantidad", ascending=False)
        # Renombrar columnas
        grouped_data.columns = ["Municipio", "Estado", "Cantidad"]
        # Calcular porcentaje sobre el total
        grouped_data["Porcentaje"] = ((grouped_data["Cantidad"] / total) * 100).round(2)
        # Reordenar columnas para mostrar Estado primero
        counts_all = grouped_data[["Estado", "Municipio", "Cantidad", "Porcentaje"]]
        
        # Obtener total de municipios únicos
        total_municipios = len(counts_all)
        
        # Obtener datos de Yucatán
        yucatan_data = counts_all[counts_all["Estado"] == "Yucatan"].reset_index(drop=True)
        
        # Inicializar estado para el primer expander si no existe
        if 'expander_municipios_abierto' not in st.session_state:
            st.session_state.expander_municipios_abierto = False
        
        # Crear expander para tablas principales con estado persistente
        with st.expander(f"Ver tabla completa de todos los Municipios y Localidades ({total_municipios})", 
                         expanded=st.session_state.expander_municipios_abierto):
            # Detectar si este expander está abierto
            if not st.session_state.expander_municipios_abierto:
                st.session_state.expander_municipios_abierto = True
            
            # Crear dos columnas
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Todos los municipios y Localidades")
                st.markdown("De Yucatan y de otros Estados ")
                st.dataframe(counts_all)
                
            with col2:
                if len(yucatan_data) > 0:
                    st.subheader(f"Municipios y Localidades de Yucatan ({len(yucatan_data)})")
                    st.dataframe(yucatan_data)
                else:
                    st.info("No hay datos para municipios/localidades en YUCATAN en el período seleccionado.")
        
        # Tablas por estado (excepto Yucatán) en un solo expander
        estados = sorted(filtro["DD_ESTADO"].unique())
        estados = [estado for estado in estados if estado != "YUCATAN"]  # Excluir Yucatán
        
        # Contar cuántas tablas de estado tendremos
        total_estados = len(estados)
        
        # Inicializar estado para el segundo expander si no existe
        if 'expander_estados_abierto' not in st.session_state:
            st.session_state.expander_estados_abierto = False
        
        # Crear expander para estados con estado persistente
        with st.expander(f"Ver municipios y Localidades por estado ({total_estados} estados)", 
                         expanded=st.session_state.expander_estados_abierto):
            # Detectar si este expander está abierto
            if not st.session_state.expander_estados_abierto:
                st.session_state.expander_estados_abierto = True
            
            # Determinar el número de columnas a mostrar (3 columnas)
            num_columnas = 3
            
            # Procesar cada estado en grupos de 3 para layout horizontal
            for i in range(0, len(estados), num_columnas):
                # Crear columnas para este grupo
                cols = st.columns(num_columnas)
                
                # Procesar los estados de este grupo (máximo 3)
                for j in range(num_columnas):
                    if i + j < len(estados):
                        estado = estados[i + j]
                        # Filtrar datos para este estado
                        estado_data = counts_all[counts_all["Estado"] == estado].reset_index(drop=True)
                        if len(estado_data) > 0:  # Solo mostrar si hay municipios
                            with cols[j]:
                                st.subheader(f"{estado} ({len(estado_data)})")
                                st.dataframe(estado_data, height=min(350, 80 + len(estado_data) * 35))
                
                # Agregar separador horizontal después de cada fila de columnas
                st.markdown("---")
    else:
        if "DD_ESTADO" not in df.columns:
            st.warning("Columnas DD_MPIO, DD_ESTADO y/o CI_FECDEN no encontradas.")
        else:
            st.warning("Columnas DD_MPIO y/o CI_FECDEN no encontradas.")

@st.fragment
def plot_estado_distribution(df: pd.DataFrame):
    """Grafica la distribución de personas desaparecidas por estado (usando DD_ESTADO)."""
    st.subheader("Distribución por Estado")
    if "DD_ESTADO" in df.columns and "CI_FECDEN" in df.columns:
        df["AÑO_DENUNCIAS"] = df["CI_FECDEN"].dt.year
        min_year = int(df["AÑO_DENUNCIAS"].min())
        max_year = int(df["AÑO_DENUNCIAS"].max())
        selected_range = st.slider("Intervalo de años (Estado) - Se toman los años en que se hicieron las denuncias", min_year, max_year, (min_year, max_year), key="slider_estado", help="Se toman los años de las denuncias")
        filtro = df[(df["AÑO_DENUNCIAS"] >= selected_range[0]) & (df["AÑO_DENUNCIAS"] <= selected_range[1])]
        counts = filtro["DD_ESTADO"].value_counts().nlargest(10)
        total = filtro["DD_ESTADO"].value_counts().sum()
        pct = (filtro["DD_ESTADO"].value_counts() / total) * 100
        fig = px.bar(x=counts.index, y=counts.values,
                     labels={"x": "Estado", "y": "Cantidad"},
                     title="Top 10 Estados con más Desapariciones")
        for i, count in enumerate(counts):
            fig.add_annotation(x=counts.index[i], y=count,
                                text=f"{count} ({pct.iloc[i]:.1f}%)", yanchor="bottom")
        st.plotly_chart(fig, use_container_width=True)

        # Tabla resumen (todos los estados) dentro de un expander
        with st.expander("Ver tabla completa de todos los estados", expanded=False):
            counts_all = filtro["DD_ESTADO"].value_counts().reset_index()
            counts_all.columns = ["Estado", "Cantidad"]
            counts_all["Porcentaje"] = (counts_all["Cantidad"] / total) * 100
            counts_all["Porcentaje"] = counts_all["Porcentaje"].round(2)
            st.dataframe(counts_all)
    else:
        st.warning("Columnas DD_ESTADO y/o CI_FECDEN no encontradas.")

@st.fragment
def plot_reapariciones(df: pd.DataFrame):
    """
    Muestra el tiempo de reaparición (días) de personas desaparecidas con filtro de meses y años.
    Opción para intervalos dinámicos (semanales, mensuales, trimestrales) y gráfica de barras, o histograma.
    """
    st.subheader("Tiempo de Reaparición")
    
    # Verificar si existen las columnas necesarias
    if "DD_FECDESAP" not in df.columns or "DL_FECLOC" not in df.columns:
        st.warning("Columnas de reaparición no encontradas. Posible problema con Base de Datos")
        return
    
    # Proceso inicial de datos - solo una vez
    # Convertir fechas y calcular tiempo de reaparición
    df["DL_FECLOC"] = pd.to_datetime(df["DL_FECLOC"], errors='coerce')
    df["TIEMPO_REAPARICION"] = (df["DL_FECLOC"] - df["DD_FECDESAP"]).dt.days
    
    # Filtrar datos inválidos de una sola vez - valores de tiempo negativos o nulos
    df_reap = df.dropna(subset=["TIEMPO_REAPARICION", "DD_FECDESAP"])
    df_reap = df_reap[df_reap["TIEMPO_REAPARICION"] >= 0]
    
    if df_reap.empty:
        st.warning("No hay datos de reaparición disponibles. Revise Etiquetas de Filtro")
        return
    
    # Extraer año y mes una sola vez para operaciones posteriores
    df_reap["YEAR"] = df_reap["DD_FECDESAP"].dt.year
    df_reap["MONTH"] = df_reap["DD_FECDESAP"].dt.month
    df_reap["PERIODO_DESAPARICION"] = df_reap["DD_FECDESAP"].dt.to_period('M')
    
    # Obtener solo los años y meses que existen en los datos
    months_years = df_reap.groupby(["YEAR", "MONTH"]).size().reset_index()
    
    # Mapeo español para nombres de meses
    month_names = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    
    # Crear opciones para el slider solo con los periodos existentes
    combined = []
    for _, row in months_years.sort_values(by=["YEAR", "MONTH"]).iterrows():
        combined.append(f"{month_names[row['MONTH']]} {row['YEAR']}")
    
    if not combined:
        st.warning("No hay periodos válidos disponibles.")
        return
    
    # Slider para seleccionar rango
    selected_range = st.select_slider("Rango de Meses y Años", combined, value=(combined[0], combined[-1]),key="rango_plot_reapariciones")
    
    # Procesar selección
    start_month, start_year = selected_range[0].split()
    end_month, end_year = selected_range[1].split()
    
    # Obtener índices numéricos
    start_month_num = list(month_names.keys())[list(month_names.values()).index(start_month)]
    end_month_num = list(month_names.keys())[list(month_names.values()).index(end_month)]
    start_year = int(start_year)
    end_year = int(end_year)
    
    # Filtrar datos por el periodo seleccionado - usar los campos ya calculados
    filtro = df_reap[
        ((df_reap["YEAR"] > start_year) | ((df_reap["YEAR"] == start_year) & (df_reap["MONTH"] >= start_month_num))) &
        ((df_reap["YEAR"] < end_year) | ((df_reap["YEAR"] == end_year) & (df_reap["MONTH"] <= end_month_num)))
    ]
    
    # Checkbox para seleccionar el tipo de visualización
    dynamic_intervals = st.checkbox("Intervalos Dinámicos y Gráfica de Barras")
    
    if filtro.empty:
        st.warning("No hay datos en el rango seleccionado.")
        return
    
    if dynamic_intervals:
        # Configuración para gráfica de barras con intervalos dinámicos
        interval_type = st.selectbox("Tipo de Intervalo", ["Semanales de 7 días", "Mensuales de 30 días", "Trimestrales de 90 días"])
        
        interval_size = 7  # Default a semanal
        if interval_type == "Mensuales de 30 días":
            interval_size = 30
        elif interval_type == "Trimestrales de 90 días":
            interval_size = 90
        
        # Calcular máximo valor considerado (8 intervalos)
        max_value = 8 * interval_size
        
        # Crear bins y etiquetas para los intervalos
        bins = [i * interval_size for i in range(9)]
        labels = [f"{bins[i]}-{bins[i+1]-1} días" for i in range(8)]
        
        # Filtrar datos para el rango y crear la categoría
        datos_rango = filtro[filtro["TIEMPO_REAPARICION"] < max_value].copy()
        
        if datos_rango.empty:
            st.warning(f"No hay datos menores a {max_value} días en el rango seleccionado.")
            return
        
        # Asignar categorías de manera eficiente
        datos_rango["RANGO_TIEMPO"] = pd.cut(
            datos_rango["TIEMPO_REAPARICION"], 
            bins=bins, 
            right=False, 
            labels=labels
        )
        
        # Contar datos por categoría de manera eficiente
        counts = datos_rango["RANGO_TIEMPO"].value_counts().reindex(labels).fillna(0).astype(int)
        
        # Calcular estadísticas
        total = len(datos_rango)
        percentages = (counts / total * 100).round(1)
        
        # Preparar etiquetas
        label_text = [f"{count} ({percentage}%)" if count > 0 else "" 
                     for count, percentage in zip(counts.values, percentages)]
        
        # Crear gráfico
        fig = px.bar(
            x=counts.index, 
            y=counts.values,
            labels={'x': 'Rango de Tiempo', 'y': 'Cantidad'},
            title=f"Distribución del Tiempo de Reaparición (8 Intervalos {interval_type})",
            text=label_text,
            color_discrete_sequence=['skyblue']#,
            #height=500
        )
        # Además, ajustar el margen superior para asegurar espacio para etiquetas
        fig.update_layout(margin=dict(t=80, b=50, l=50, r=50))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Información sobre datos fuera del rango
        datos_fuera = filtro[filtro["TIEMPO_REAPARICION"] >= max_value]
        if not datos_fuera.empty:
            porcentaje_fuera = round((len(datos_fuera) / len(filtro) * 100), 1)
            st.info(f"Nota: {len(datos_fuera)} registros ({porcentaje_fuera}%) tienen un tiempo de reaparición mayor a {max_value} días y no se muestran en la gráfica.")
    
    else:
        # Histograma original para tiempos menores a 30 días
        max_days_for_histogram = 29
        datos_histograma = filtro[filtro["TIEMPO_REAPARICION"] <= max_days_for_histogram]
        
        if datos_histograma.empty:
            st.warning(f"No hay datos menores a {max_days_for_histogram+1} días en el rango seleccionado.")
            return
        
        # Calcular el histograma de manera eficiente
        counts, bins = np.histogram(datos_histograma["TIEMPO_REAPARICION"], bins=30, range=(0, max_days_for_histogram+1))
        
        # Crear la figura con datos pre-calculados
        fig = px.histogram(
            datos_histograma, 
            x="TIEMPO_REAPARICION", 
            nbins=30,
            range_x=[0, max_days_for_histogram+1],
            labels={"TIEMPO_REAPARICION": "Días"},
            title="Distribución del Tiempo de Reaparición (hasta 30 días)"
        )
        # Actualizar explícitamente el nombre del eje y
        fig.update_layout(yaxis_title="Cantidad",
                          margin = dict(t=80,b=50,l=50,r=50))
        
        # Agregar anotaciones eficientemente
        total_data_points = len(datos_histograma)
        for i in range(len(counts)):
            bin_center = (bins[i] + bins[i+1]) / 2
            if counts[i] > 0:  # Solo agregar anotaciones para barras con datos
                percentage = (counts[i] / total_data_points) * 100
                fig.add_annotation(
                    x=bin_center, 
                    y=counts[i],
                    text=f"{counts[i]}", 
                    yanchor="top", 
                    yshift=0
                )
                fig.add_annotation(
                    x=bin_center, 
                    y=counts[i],
                    text=f"{percentage:.1f}%", 
                    yanchor="bottom", 
                    yshift=25, 
                    showarrow=False
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Información sobre datos excluidos
        datos_excluidos = filtro[filtro["TIEMPO_REAPARICION"] > max_days_for_histogram]
        if not datos_excluidos.empty:
            porcentaje_excluidos = round((len(datos_excluidos) / len(filtro) * 100), 1)
            st.info(f"Nota: {len(datos_excluidos)} registros ({porcentaje_excluidos}%) tienen un tiempo de reaparición mayor a {max_days_for_histogram} días y no se muestran en el histograma.")

# ---------------------- ANALISIS DE CONTEXTO---------------------- #
# ---------------------------
# Configuración global
# ---------------------------
CATEGORIAS_PREDEFINIDAS = {
    'CON_PAREJA': {
        'grupo': 'TIPO_DESAPARICION',
        'palabras_clave': {
            'fuga': ['FUG', 'ESCAP', 'FUE CON', 'SE FUE', 'HUY', 'ESTA', 'IBA CON', 'IRIA CON'],
            'pareja': ['NOVIO', 'PAREJA', 'NOVIA', 'AMANTE', 'ENAMORAD', 'QUERID']
        }
    },
    'CON_AMIGOS': {
        'grupo': 'TIPO_DESAPARICION',
        'palabras_clave': {
            'fuga': ['FUG', 'ESCAP', 'FUE CON', 'SE FUE', 'HUY', 'ESTA', 'IBA CON', 'IRIA CON'],
            'amistad': ['AMIG', 'AMISTAD']
        }
    },
    'CONYUGUE': {
        'grupo': 'TIPO_DESAPARICION',
        'palabras_clave': {
            'pareja': ['ESPOS', 'CONYUGUE', 'MARIDO']
        }
    },
    'FUGA_DE_ALBERGUE': {
        'grupo': 'TIPO_DESAPARICION',
        'palabras_clave': {
            'fuga': ['FUG', 'ESCAP', 'SALIO', 'NO REGRES', 'SE FUE', 'HUY'],
            'institucion': ['ALBERGUE', 'HORFANATO', 'CAIMEDE', 'REFUGIO', 'HOGAR', 'CONVENTO', 'DIF ', 'CRIA ']
        }
    },
    'INGRESO_ALBERGUE': {
        'grupo': 'TIPO_DESAPARICION_POSIBLE_CAUSA',
        'palabras_clave': {
            'fuga': ['ENTR', 'INGRES', 'INTERN'],
            'institucion': ['ALBERGUE', 'HORFANATO', 'CAIMEDE', 'REFUGIO', 'HOGAR', 'CONVENTO', 'DIF ', 'CRIA ']
        }
    },
    'SALIO_DE_CASA_NO_VOLVIO': {
        'grupo': 'HECHOS',
        'palabras_clave': {
            'fuga': ['FUG', 'ESCAP', 'SALIO', 'NO REGRES', 'SE FUE', 'HUY'],
            'casa': ['CASA', 'DOMICILIO', 'FAMILIA', 'ADOLESCENTE']
        }
    },
    'ADOLESCENTE_ESCAPA': {
        'grupo': 'HECHOS',
        'palabras_clave': {
            'casa': ['ADOLESCENTE QUE SE ESCAPA']
        }
    },
    'DISCUSION': {
        'grupo': 'HECHOS_PREVIOS_POSIBLE_CAUSA',
        'palabras_clave': {
            'problema': ['DISCU', 'PELE', 'GRIT', 'ARGUMENTO', 'PROBLEMA', 'MALTRAT', 'VIOLENCIA']
        }
    },
    'CRISIS_MENTAL': {
        'grupo': 'HECHOS_PREVIOS_POSIBLE_CAUSA',
        'palabras_clave': {
            'crisis': ['CRISIS', 'DEPRESION', 'NERVIOS', 'ANSIEDAD', 'TRASTORNO', 'ALZHEIMER', 'DEMENCIA', 'MENTAL', 'PADEC']
        }
    },
    'CENTRO_REHABILITACION': {
        'grupo': 'TIPO_DESAPARICION',
        'palabras_clave': {
            'centro': ['REHABILITACION', 'ANEXO']
        }
    },
    'ESTABA_PADRE_MADRE': {
        'grupo': 'HECHOS',
        'palabras_clave': {
            'llevar': ['ESTA CON SU', 'ESTABA CON'],
            'familia': ['SU PAPA', 'SU PADRE', 'SU MAMA', 'SU MADRE']
        }
    },
    'LLEVADO_POR_PADRE_MADRE': {
        'grupo': 'TIPO_DESAPARICION',
        'palabras_clave': {
            'llevar': ['SE LA LLEV', 'SE LO LLEV', 'SE LOS LLEV', 'SE LAS LLEV', 'CON SU HIJ', 'CON SUS HIJ', 'NIÑOS'],
            'familia': ['SU PAPA', 'SU PADRE', 'PAREJA', 'SU MAMA', 'SU MADRE', 'ELLA']
        }
    },
    'ABANDONO_HOGAR': {
        'grupo': 'TIPO_DESAPARICION',
        'palabras_clave': {
            'casa': ['ABANDONO DE HOGAR']
        }
    },
    'OTRO_ESTADO': {
        'grupo': 'HECHOS',
        'palabras_clave': {
            'estado': ["AGUASCALIENTES", "BAJA CALIFORNIA", "BAJA CALIFORNIA SUR", "CAMPECHE", "COAHUILA", "COLIMA", "CHIAPAS",
                       "CHIHUAHUA", "CIUDAD DE MEXICO", "DURANGO", "GUANAJUATO", "GUERRERO", "HIDALGO", "JALISCO", "ESTADO DE MEXICO",
                       "MICHOACAN", "MORELOS", "NAYARIT", "NUEVO LEON", "OAXACA", "PUEBLA", "QUERETARO", "QUINTANA ROO", "SAN LUIS POTOSI",
                       "SINALOA", "SONORA", "TABASCO", "TAMAULIPAS", "TLAXCALA", "VERACRUZ", "ZACATECAS"]
        }
    },
    'ACTIVIDAD': {
        'grupo': 'HECHOS_PREVIOS',
        'palabras_clave': {
            'actividad': ['SALIO A', 'FUE A', 'IRIA A', 'IBA A', 'RUMBO A', 'VA A', 'TRABAJ', 'ESTUDIAR', 'ESCUELA', 'UNIVERSIDAD', 'ACTIVIDAD', 'MONTAR', 'CORRER']
        }
    },
    'DETENCION': {
        'grupo': 'TIPO_DESAPARICION_POSIBLE_CAUSA',
        'palabras_clave': {
            'actividad': ['ESTAB', 'FUE', 'ESTUVO', 'IBA', 'LLEVAN', 'ENCONTRABA', 'DELITO'],
            'detencion': ['DETENID', 'ARRESTAD', 'RECLUID', 'CERESO']
        }
    },
    'EBRIEDAD_DROGAS': {
        'grupo': 'HECHOS_PREVIOS_POSIBLE_CAUSA',
        'palabras_clave': {
            'sustancias': ['ALCOHOL', 'EBRI', 'BORRACH', 'EMBRIAG', 'BEBID', 'TOMANDO', 'ALCO',
                            'DROGA', 'DROGAD', 'SUSTANCIA', 'ESTUPEFACIENTE', 'INTOXICA', 'BEBIENDO',
                            'MARIHUANA', 'COCAINA', 'PASTILLAS', 'CRACK', 'METANFETAMINA']
        }
    }
}

GRUPOS_CATEGORIAS = {info['grupo'] for info in CATEGORIAS_PREDEFINIDAS.values()}
COLORES_GRUPOS = {
    'TIPO_DESAPARICION': '#4E79A7',
    'TIPO_DESAPARICION_POSIBLE_CAUSA': '#F28E2B',
    'HECHOS': '#E15759',
    'HECHOS_PREVIOS': '#76B7B2',
    'HECHOS_PREVIOS_POSIBLE_CAUSA': '#59A14F'
}

# ---------------------------
# Funciones de Procesamiento
# ---------------------------
def categorizar_registros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de categorías y grupos a partir de DD_HECHOS.
    """
    if df is None or df.empty:
        return None

    df_categorizado = df.copy()
    df_categorizado['DD_HECHOS'] = df_categorizado['DD_HECHOS'].astype(str).str.upper()

    # Inicializar columnas de categorías y grupos
    for categoria in CATEGORIAS_PREDEFINIDAS.keys():
        df_categorizado[categoria] = 0
    for grupo in GRUPOS_CATEGORIAS:
        df_categorizado[f'GRUPO_{grupo}'] = 0

    # Procesar cada registro
    for idx, row in df_categorizado.iterrows():
        descripcion = row['DD_HECHOS']
        for categoria, info in CATEGORIAS_PREDEFINIDAS.items():
            palabras_clave = info['palabras_clave']
            if all(any(re.search(r'\b' + palabra, descripcion) for palabra in lista)
                   for lista in palabras_clave.values()):
                df_categorizado.at[idx, categoria] = 1
                df_categorizado.at[idx, f'GRUPO_{info["grupo"]}'] = 1

    df_categorizado['TOTAL_CATEGORIAS'] = df_categorizado[list(CATEGORIAS_PREDEFINIDAS.keys())].sum(axis=1)
    df_categorizado['CATEGORIAS_DETECTADAS'] = df_categorizado.apply(
        lambda row: ', '.join([cat for cat in CATEGORIAS_PREDEFINIDAS.keys() if row[cat] == 1]) or 'SIN_CATEGORIA',
        axis=1
    )
    return df_categorizado

# ---------------------------
# Funciones de Visualización Interactiva con Plotly
# ---------------------------
def plot_resumen_categorias(df_categorizado: pd.DataFrame):
    resumen = df_categorizado[list(CATEGORIAS_PREDEFINIDAS.keys())].sum().sort_values(ascending=False).reset_index()
    resumen.columns = ['Categoria', 'Total']
    resumen['Grupo'] = resumen['Categoria'].apply(lambda cat: CATEGORIAS_PREDEFINIDAS[cat]['grupo'])
    fig = px.bar(resumen, x='Categoria', y='Total', color='Grupo',
                 text='Total', color_discrete_map=COLORES_GRUPOS)
    fig.update_layout(title="Distribución de registros por categoría", xaxis_tickangle=-45)
    return fig

def plot_resumen_por_grupo(df_categorizado: pd.DataFrame):
    figs = {}
    for grupo in GRUPOS_CATEGORIAS:
        cats = [cat for cat, info in CATEGORIAS_PREDEFINIDAS.items() if info['grupo'] == grupo]
        if not cats:
            continue
        data = pd.DataFrame({
            'Categoria': cats,
            'Total': [df_categorizado[cat].sum() for cat in cats]
        })
        fig = px.bar(data, x='Categoria', y='Total', text='Total',
                     color_discrete_sequence=[COLORES_GRUPOS[grupo]])
        fig.update_layout(title=f"Registros por categoría - Grupo: {grupo}", xaxis_tickangle=-45)
        figs[grupo] = fig
    return figs

def plot_comparativa_grupos(df_categorizado: pd.DataFrame):
    conteo = {grupo: df_categorizado[f'GRUPO_{grupo}'].sum() for grupo in GRUPOS_CATEGORIAS}
    data = pd.DataFrame({
        'Grupo': list(conteo.keys()),
        'Total': list(conteo.values())
    })
    fig = px.bar(data, x='Grupo', y='Total', text='Total', color='Grupo',
                 color_discrete_map=COLORES_GRUPOS)
    fig.update_layout(title="Comparativa de registros por grupo de categorías", xaxis_tickangle=-45)
    return fig

def plot_distribucion_sin_categoria(df_categorizado: pd.DataFrame):
    con_cat = (df_categorizado['TOTAL_CATEGORIAS'] > 0).sum()
    sin_cat = (df_categorizado['TOTAL_CATEGORIAS'] == 0).sum()
    data = pd.DataFrame({
        'Estado': ['Con categoría', 'Sin categoría'],
        'Total': [con_cat, sin_cat]
    })
    fig = px.pie(data, names='Estado', values='Total',
                 title="Distribución de registros categorizados vs no categorizados",
                 color_discrete_sequence=['#66b3ff', '#ff9999'])
    return fig

def plot_heatmap_categorias_por_grupo(df_categorizado: pd.DataFrame):
    # Crear matriz: filas=grupo, columnas=categoría
    data = {}
    for grupo in GRUPOS_CATEGORIAS:
        row = {}
        for cat, info in CATEGORIAS_PREDEFINIDAS.items():
            if info['grupo'] == grupo:
                row[cat] = df_categorizado[cat].sum()
        data[grupo] = row
    df_heatmap = pd.DataFrame(data).T
    fig = px.imshow(df_heatmap, text_auto=True, aspect="auto", color_continuous_scale="YlGnBu",
                    title="Mapa de calor: Distribución de categorías por grupo")
    return fig

def mostrar_ejemplos_por_categoria(df_categorizado: pd.DataFrame, num_ejemplos=3):
    st.markdown("### Ejemplos de registros por categoría")
    # Organiza las categorías por grupo
    categorias_por_grupo = {}
    for cat, info in CATEGORIAS_PREDEFINIDAS.items():
        categorias_por_grupo.setdefault(info['grupo'], []).append(cat)

    for grupo, categorias in categorias_por_grupo.items():
        st.markdown(f"#### Grupo: {grupo}")
        for cat in categorias:
            registros = df_categorizado[df_categorizado[cat] == 1]
            if not registros.empty:
                st.markdown(f"**Categoría: {cat}** (Total: {len(registros)} registros)")
                muestra = registros.sample(min(num_ejemplos, len(registros)))
                for i, (_, row) in enumerate(muestra.iterrows(), 1):
                    desc = row['DD_HECHOS']
                    if len(desc) > 300:
                        desc = desc[:300] + "..."
                    st.markdown(f"*Ejemplo {i}:* {desc}  \n_Otras categorías:_ {row['CATEGORIAS_DETECTADAS']}")
            else:
                st.markdown(f"**Categoría: {cat}** - No se encontraron registros.")


# -------------------------
# Tabla Registro Estatal (Denuncias)
# -------------------------
def generate_registro_estatal_table_denuncias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera la tabla de Registro Estatal usando las fechas de denuncia (CI_FECDEN)
    con las columnas: AÑO, PERSONAS REPORTADAS, PERSONAS LOCALIZADAS, PENDIENTES.
    Se considera únicamente datos desde el año 2001 hasta el año actual y se agrega
    la fila TOTAL al final.
    """
    if "CI_FECDEN" not in df.columns or "PD_ESTATUSVIC" not in df.columns:
        st.warning("Columnas necesarias no encontradas para la tabla de denuncias.")
        return pd.DataFrame()
    
    df["AÑO_DENUNCIAS"] = df["CI_FECDEN"].dt.year
    df_den = df[df["AÑO_DENUNCIAS"] >= 2001]
    
    tabla = (df_den.groupby("AÑO_DENUNCIAS")
             .agg(PERSONAS_REPORTADAS=("AÑO_DENUNCIAS", "count"),
                  PERSONAS_LOCALIZADAS=("PD_ESTATUSVIC", lambda x: (x == "LOCALIZADA").sum()))
             .reset_index()
             .rename(columns={"AÑO_DENUNCIAS": "AÑO"}))
    
    # Convertir la columna AÑO a string para evitar problemas de conversión
    tabla["AÑO"] = tabla["AÑO"].astype(str)
    tabla["PENDIENTES"] = tabla["PERSONAS_REPORTADAS"] - tabla["PERSONAS_LOCALIZADAS"]
    
    totales = pd.DataFrame({
        "AÑO": ["TOTAL"],
        "PERSONAS_REPORTADAS": [tabla["PERSONAS_REPORTADAS"].sum()],
        "PERSONAS_LOCALIZADAS": [tabla["PERSONAS_LOCALIZADAS"].sum()],
        "PENDIENTES": [tabla["PENDIENTES"].sum()]
    })
    
    tabla = pd.concat([tabla, totales], ignore_index=True)
    return tabla


def style_registro_estatal(df: pd.DataFrame) -> str:
    """Devuelve la tabla estilizada en HTML sin índice."""
    styled = (df.style
              .apply(lambda row: ['background-color: #0000FF; color: white' if row['AÑO'] == 'TOTAL' else '' for _ in row],
                     axis=1)
              .set_table_styles([{'selector': 'th', 'props': [('background-color', 'lightblue'), ('color', 'black')]}])
              .hide(axis="index"))
    return styled.to_html()

def formato_fecha_actual() -> str:
    """Devuelve la fecha actual en el formato '08 de Febrero 2025'."""
    hoy = datetime.today()
    meses = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
             7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    return hoy.strftime("%d") + " de " + meses[hoy.month] + " " + hoy.strftime("%Y")

# -------------------------
# Shutdown_Signal
# -------------------------

# Ruta para el archivo de señal
signal_file = "shutdown_signal.txt"

# Función para verificar la señal
def check_shutdown_signal():
    while True:
        if os.path.exists(signal_file):
            # Elimina el archivo de señal
            os.remove(signal_file)
            print("Cerrando aplicación Streamlit...")
            # Envía señal SIGINT (equivalente a Ctrl+C)
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)
            return
        time.sleep(1)


# -------------------------
# Funciones para cada pestaña
# -------------------------
def tab_desapariciones(data):
    """Contenido de la pestaña Desapariciones."""
    st.header("Desapariciones")
    
    total_desaparecidos, total_localizados = display_metrics(data)
    st.markdown("---")
    
    plot_desapariciones_por_año(data)
    
    st.write("---")
    st.subheader("🔮 Predicción de Personas Desaparecidas con ARIMA")
    st.write("Visualización de la predicción con intervalos de confianza")

    training_series, año_actual_valor_actual = procesar_datos_ARIMA(data)
    training_series = training_series.to_frame().reset_index()
    col1, col2 = st.columns(2)
    with col1:
        confidence_level = st.slider("Nivel de Confianza (%)", 50, 99, 95, 1)
    with col2:
        years_to_predict = st.slider("Años a Predecir", 1, 10, 5, 1)

    if not training_series.empty:
        modelo_arima = entrenar_modelo_arima(training_series)
        predict = generar_prediccion_arima(modelo_arima, training_series, years_to_predict, confidence_level / 100)
        mostrar_prediccion(predict, training_series, confidence_level, año_actual_valor_actual)
    
    st.markdown("---")
    plot_denuncias_por_año(data)
    plot_expedientes_por_año(data)
    
    st.markdown("---")
    #Registro Estatal (Denuncias)
    fecha_max_denuncia = data["CI_FECDEN"].max().date()
    tabla_registro = generate_registro_estatal_table_denuncias(data)
    if not tabla_registro.empty:
        st.subheader(f"Registro Estatal (Denuncias) corte al {fecha_max_denuncia} en fecha {formato_fecha_actual()}")
        st.markdown(style_registro_estatal(tabla_registro), unsafe_allow_html=True)
        #tabla_registro.set_index("AÑO", inplace=True)
        #st.dataframe(tabla_registro)
    
    st.write("---")
    validar_fechas_desaparicion(data)

def tab_reapariciones(data):
    """Contenido de la pestaña Reapariciones."""
    st.header("Reapariciones")
    display_metrics_reaparecidos(data)
    st.markdown("---")
    plot_reapariciones(data)
    plot_outliers_reapariciones(data)

def tab_demograficos(data):
    """Contenido de la pestaña Demográficos."""
    st.header("Demográficos")
    plot_sexo_distribution(data)
    plot_clasedad_distribution(data)
    plot_age_distribution(data)
    st.markdown("---")
    plot_municipio_distribution(data)
    plot_estado_distribution(data)
    st.markdown("---")

def tab_analisis_contexto(data: pd.DataFrame):
    st.header("Análisis de Categorías y Contexto")
    
    # Filtrar por año si existe la columna de fecha de desaparición
    if "CI_FECDEN" in data.columns:
        #data["DD_FECDESAP"] = pd.to_datetime(data["DD_FECDESAP"], errors='coerce')
        data = data.dropna(subset=["CI_FECDEN"])
        data["AÑO_DENUNCIA"] = data["CI_FECDEN"].dt.year
        min_year = int(data["AÑO_DENUNCIA"].min())
        max_year = int(data["AÑO_DENUNCIA"].max())
        selected_range = st.slider("Analisis Contexto (Denuncias)", min_year, max_year, (min_year, max_year))
        data_filtered = data[(data["AÑO_DENUNCIA"] >= selected_range[0]) & 
                             (data["AÑO_DENUNCIA"] <= selected_range[1])]
    else:
        st.warning("La columna 'CI_FECDEN' no se encontró. Se usarán todos los datos.")
        data_filtered = data

    # Procesar y categorizar registros
    df_categorizado = categorizar_registros(data_filtered)
    if df_categorizado is None:
        st.error("No se pudieron categorizar los registros.")
        return

    st.subheader("Resumen de Categorías")
    st.plotly_chart(plot_resumen_categorias(df_categorizado), use_container_width=True)

    st.subheader("Resumen por Grupo")
    figs_por_grupo = plot_resumen_por_grupo(df_categorizado)
    for grupo, fig in figs_por_grupo.items():
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparativa de Grupos")
    st.plotly_chart(plot_comparativa_grupos(df_categorizado), use_container_width=True)

    st.subheader("Distribución con/ sin Categoría")
    st.plotly_chart(plot_distribucion_sin_categoria(df_categorizado), use_container_width=True)

    st.subheader("Heatmap de Categorías por Grupo")
    st.plotly_chart(plot_heatmap_categorias_por_grupo(df_categorizado), use_container_width=True)

    mostrar_ejemplos_por_categoria(df_categorizado, num_ejemplos=3)

# -------------------------
# Función principal
# -------------------------
def main():
    """Función principal de la aplicación."""
    setup_page_config()
    st.title("📊 :blue[Monitoreo de Personas Desaparecidas]")

    # Cargar y limpiar datos (con caché)
    default_filepath = r"C:\Users\Angel\Documents\Programacion\Fiscalia\data_desaparecidos.xlsx"

    ###################################
    ## ------   SUBIR EXCEL   ------ ##
    
    # Opción de subir archivo
    st.sidebar.markdown("<h1 style='text-align: center;'>Carga de Datos</h1>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(":violet[Subir archivo Excel]", type=['xlsx', 'xls'])
    
    # Cargar datos (con caché para optimizar)
    data, archivo_actual = load_data(
        uploaded_file=uploaded_file, 
        default_filepath=default_filepath
    )

    # Mostrar información del dataset en la barra lateral
    if archivo_actual:
        st.sidebar.info(f"Archivo actual: {archivo_actual}")
    ###################################
    
    if data is None: return
    data = clean_data(data)
    data_complete = data.copy()
    
    
    # Aplicar filtros en la barra lateral
    st.sidebar.markdown("<h1 style='text-align: center;'>Filtros</h1>", unsafe_allow_html=True)
    # Inicializar variables de estado para controlar los widgets
    if 'filter_key' not in st.session_state:
        st.session_state.filter_key = 0
    data_filtered = create_date_filter(data)
    

    ####################################################
    # Aplicar filtros de selección múltiple

    # Crear un key único para cada widget basado en filter_key
    current_key = st.session_state.filter_key

    # Filtro por estatus (denuncias) con key dinámico
    estados = st.sidebar.multiselect("Seleccione el :violet[Estatus] de las Personas",
                                      options=data_complete["PD_ESTATUSVIC"].unique(),
                                      default=data_complete["PD_ESTATUSVIC"].unique().tolist(),
                                      key=f"estados_{current_key}")
    data_filtered = data_filtered[data_filtered["PD_ESTATUSVIC"].isin(estados)]

    # Filtro por sexo (denuncias) con key dinámico
    sexo = st.sidebar.multiselect("Seleccione el :violet[Sexo] de las Personas",
                                      options=data_complete["PD_SEXO"].unique(),
                                      default=data_complete["PD_SEXO"].unique().tolist(),
                                      key=f"sexo_{current_key}")
    data_filtered = data_filtered[data_filtered["PD_SEXO"].isin(sexo)]

    # Filtro por clase de edad (denuncias) con key dinámico
    clases_edad = st.sidebar.multiselect("Seleccione la :violet[Clase] de Edad de Personas",
                                      options=data_complete["PD_CLASEDAD"].unique(),
                                      default=data_complete["PD_CLASEDAD"].unique().tolist(),
                                      key=f"edad_{current_key}")
    data_filtered = data_filtered[data_filtered["PD_CLASEDAD"].isin(clases_edad)]
    
    if data_filtered.empty:
        st.error("No hay datos para los filtros seleccionados.")
        return
    
    ###################################

    # Inicia el hilo de verificación (una sola vez)
    if "shutdown_thread_started" not in st.session_state:
        shutdown_thread = threading.Thread(target=check_shutdown_signal)
        shutdown_thread.daemon = True
        shutdown_thread.start()
        st.session_state.shutdown_thread_started = True
    
    # Inicializar estados si no existen
    if "show_confirm" not in st.session_state:
        st.session_state.show_confirm = False
    
    # Botón para mostrar confirmación###############
    #if st.sidebar.button("Cerrar Aplicación"):
    #    st.session_state.show_confirm = True
    
    # Mostrar confirmación si el estado lo requiere
    if st.session_state.show_confirm:
        confirm = st.sidebar.checkbox("¿Estás seguro?")
        if confirm:
            with open(signal_file, "w") as f:
                f.write("shutdown")
            st.sidebar.success("Cerrando la aplicación...")
            st.stop()  # Detiene la ejecución del script actual
    ###################################
    # Crear pestañas
    tabs = st.tabs(["Desapariciones", "Reapariciones", "Demográficos", "Analisís de Contexto"])
    
    # Contenido de cada pestaña
    with tabs[0]:
        tab_desapariciones(data_filtered)
    
    with tabs[1]:
        tab_reapariciones(data_filtered)
    
    with tabs[2]:
        tab_demograficos(data_filtered)
        
    with tabs[3]:
        tab_analisis_contexto(data_filtered)

if __name__ == '__main__':
    main()
