import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error, r2_score
import joblib # Para guardar y cargar modelos

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Análisis y Predicción de Ventas de Videojuegos",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎮 Análisis y Predicción de Ventas de Videojuegos")
st.markdown("Esta aplicación permite explorar un dataset de ventas de videojuegos y predecir categorías de ventas globales usando modelos de clasificación.")

# --- Carga de Datos ---
@st.cache_data # Cachea los datos para no recargarlos cada vez
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return None
    return None

st.sidebar.header("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV (por ejemplo, video_games_sales.csv)", type=["csv"])

df = load_data(uploaded_file)

if df is not None:
    st.sidebar.success("¡Dataset cargado exitosamente!")
    st.subheader("Vista Previa del Dataset")
    st.dataframe(df.head())

    # --- Preprocesamiento de Datos ---
    st.sidebar.header("Preprocesamiento")

    # Eliminación de columnas irrelevantes/con muchos nulos para el análisis de clasificación
    cols_to_drop = ['Developer', 'Rating'] # 'Developer' y 'Rating' pueden tener demasiadas categorías únicas o nulos
    if 'Critic_Score' in df.columns:
        df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')
    if 'User_Score' in df.columns:
        df['User_Score'] = df['User_Score'].replace('tbd', np.nan)
        df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    if 'Year_of_Release' in df.columns:
        df['Year_of_Release'] = pd.to_numeric(df['Year_of_Release'], errors='coerce')

    # Eliminar filas con valores nulos en columnas críticas para la predicción
    df_cleaned = df.dropna(subset=['Year_of_Release', 'Genre', 'Publisher', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Global_Sales']).copy()

    # Convertir 'User_Score' a float
    df_cleaned['User_Score'] = df_cleaned['User_Score'].astype(float)

    st.sidebar.write(f"Filas originales: {len(df)}")
    st.sidebar.write(f"Filas después de limpieza de nulos: {len(df_cleaned)}")

    # --- Discretización de Global_Sales ---
    st.subheader("Discretización de Ventas Globales")
    num_bins = st.sidebar.slider("Número de categorías para Ventas Globales", min_value=2, max_value=20, value=12)

    # Discretizar Global_Sales en 'num_bins' cuantiles
    df_cleaned['Global_Sales_Category'] = pd.qcut(df_cleaned['Global_Sales'], q=num_bins, labels=False, duplicates='drop')
    st.write(f"Se crearon {num_bins} categorías para las ventas globales.")
    st.write(df_cleaned['Global_Sales_Category'].value_counts().sort_index())
    st.write(f"Rangos de ventas por categoría (para {num_bins} cuantiles):")
    # Mostrar los rangos reales de los cuantiles
    min_max_sales_per_category = df_cleaned.groupby('Global_Sales_Category')['Global_Sales'].agg(['min', 'max'])
    st.dataframe(min_max_sales_per_category)
    st.info(f"**Interpretación de la Clase 0:** Corresponde al rango de ventas globales más bajo: {min_max_sales_per_category.loc[0, 'min']:.2f} - {min_max_sales_per_category.loc[0, 'max']:.2f} millones.")
    st.info(f"**Interpretación de la Clase 1:** Corresponde al segundo rango de ventas globales más bajo: {min_max_sales_per_category.loc[1, 'min']:.2f} - {min_max_sales_per_category.loc[1, 'max']:.2f} millones.")


    # Codificación de variables categóricas
    st.sidebar.subheader("Codificación Categórica")
    categorical_cols = ['Genre', 'Publisher'] # Añade más si consideras relevantes
    for col in categorical_cols:
        if col in df_cleaned.columns:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df_cleaned[col])
            st.sidebar.write(f"'{col}' codificado con LabelEncoder.")

    # Variables numéricas para escalado
    numeric_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Year_of_Release']
    scaler = StandardScaler()
    df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
    st.sidebar.write("Variables numéricas escaladas con StandardScaler.")

    # --- Selección de Características y División de Datos ---
    st.sidebar.header("Configuración del Modelo")
    features = st.sidebar.multiselect(
        "Selecciona las características (X) para el modelo:",
        options=df_cleaned.drop(columns=['Name', 'Global_Sales', 'Global_Sales_Category'], errors='ignore').columns.tolist(),
        default=[col for col in df_cleaned.drop(columns=['Name', 'Global_Sales', 'Global_Sales_Category'], errors='ignore').columns if col not in ['Developer', 'Rating']]
    )

    if not features:
        st.error("Por favor, selecciona al menos una característica.")
    else:
        X = df_cleaned[features]
        y = df_cleaned['Global_Sales_Category']

        test_size = st.sidebar.slider("Tamaño del conjunto de prueba (Test Size):", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.sidebar.number_input("Semilla aleatoria (Random State):", value=42, step=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        st.sidebar.write(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
        st.sidebar.write(f"Tamaño del conjunto de prueba: {len(X_test)}")

        # --- Selección y Entrenamiento del Modelo ---
        model_choice = st.sidebar.selectbox(
            "Selecciona el modelo de clasificación:",
            options=['Árbol de Decisión', 'K-Nearest Neighbors (KNN)']
        )

        st.subheader(f"Entrenamiento y Evaluación del Modelo: {model_choice}")

        if model_choice == 'Árbol de Decisión':
            max_depth = st.sidebar.slider("Profundidad máxima del árbol (max_depth):", min_value=2, max_value=20, value=10)
            dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            dt_classifier.fit(X_train, y_train)
            y_pred = dt_classifier.predict(X_test)

            st.pyplot(plt.figure(figsize=(20,10)))
            plot_tree(dt_classifier, feature_names=X.columns, class_names=[str(i) for i in range(num_bins)], filled=True, rounded=True)
            st.pyplot(plt) # Muestra el árbol de decisión

            model_name = "decision_tree_model.joblib"
            joblib.dump(dt_classifier, model_name)
            st.sidebar.download_button(
                label="Descargar Modelo de Árbol de Decisión",
                data=open(model_name, "rb").read(),
                file_name=model_name,
                mime="application/octet-stream"
            )

        elif model_choice == 'K-Nearest Neighbors (KNN)':
            n_neighbors = st.sidebar.slider("Número de vecinos (n_neighbors):", min_value=1, max_value=30, value=5)
            knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn_classifier.fit(X_train, y_train)
            y_pred = knn_classifier.predict(X_test)

            model_name = "knn_classifier_model.joblib"
            joblib.dump(knn_classifier, model_name)
            st.sidebar.download_button(
                label="Descargar Modelo KNN",
                data=open(model_name, "rb").read(),
                file_name=model_name,
                mime="application/octet-stream"
            )

        # --- Métricas de Evaluación ---
        st.subheader("Métricas de Evaluación")

        accuracy = dt_classifier.score(X_test, y_test) if model_choice == 'Árbol de Decisión' else knn_classifier.score(X_test, y_test)
        st.write(f"**Accuracy (Precisión):** {accuracy:.4f}")

        st.subheader("Reporte de Clasificación")
        st.text(classification_report(y_test, y_pred, target_names=[f'Clase {i}' for i in range(num_bins)], zero_division=0))

        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred, labels=range(num_bins))
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Clase {i}' for i in range(num_bins)])
        disp.plot(cmap='Blues', values_format='d', ax=ax)
        ax.set_title(f'Matriz de Confusión - {model_choice}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)

        st.markdown("""
        **Interpretación de la Matriz de Confusión:**
        * **Eje Y (True label):** Representa las categorías de ventas globales reales.
        * **Eje X (Predicted label):** Representa las categorías de ventas globales predichas por el modelo.
        * **Diagonal Principal:** Muestra el número de predicciones correctas para cada categoría.
        * **Fuera de la Diagonal:** Muestra los errores de clasificación (instancias que pertenecen a una categoría real pero fueron predichas como otra).
        """)

else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis.")

st.sidebar.markdown("---")
st.sidebar.info("Desarrollado con Streamlit y Scikit-learn.")