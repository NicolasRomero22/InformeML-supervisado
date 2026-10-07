# ============================================================
# PRÁCTICA: MACHINE LEARNING SUPERVISADO - CASO RECURSOS HUMANOS
# Predicción de rotación (renuncia) de empleados
# Algoritmos: Regresión Logística, Árbol de Decisión, Random Forest,
#             SVM, KNN y Naive Bayes
# ============================================================

# ------------------------------------------------------------
# CELDA 1. LIBRERÍAS
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # En Colab puede omitirse; permite guardar figuras sin pantalla
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ------------------------------------------------------------
# CELDA 2. DATASET
# renuncia: 0 = permanece | 1 = renuncia
# ------------------------------------------------------------
data = {
    "edad": [25, 28, 35, 40, 30, 45, 32, 29, 50, 38,
             27, 33, 41, 36, 31, 48, 26, 39, 42, 34],
    "salario": [1800, 2000, 3500, 4500, 2200, 5000, 2800, 2100, 6000, 4000,
                1900, 3000, 4700, 3600, 2500, 5500, 1700, 4200, 4800, 3200],
    "años_empresa": [1, 2, 5, 8, 2, 10, 4, 2, 15, 7,
                     1, 4, 9, 6, 3, 12, 1, 8, 10, 5],
    "satisfaccion": [2, 3, 7, 8, 3, 9, 5, 2, 9, 7,
                     2, 6, 8, 6, 4, 9, 1, 7, 8, 5],
    "horas_extra": [15, 12, 5, 4, 14, 3, 8, 13, 2, 6,
                    16, 7, 4, 6, 10, 3, 18, 5, 4, 8],
    "renuncia": [1, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
}
df = pd.DataFrame(data)

print("=" * 70)
print("1. EXPLORACIÓN DEL DATASET")
print("=" * 70)
print("\nPrimeras filas:")
print(df.head())
print("\nDimensiones (filas, columnas):", df.shape)
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nEstadísticas descriptivas:")
print(df.describe().round(2))
print("\nDistribución de la variable objetivo (renuncia):")
print(df["renuncia"].value_counts().rename({0: "Permanece", 1: "Renuncia"}))

# Promedios por clase
print("\nPromedio de cada variable según la clase:")
print(df.groupby("renuncia").mean().round(2))

# Figura 1: distribución de clases y matriz de correlación
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
conteo = df["renuncia"].value_counts().sort_index()
ax[0].bar(["Permanece (0)", "Renuncia (1)"], conteo.values,
          color=["#2e7d32", "#c62828"])
ax[0].set_title("Distribución de la variable objetivo")
ax[0].set_ylabel("Número de empleados")
for i, v in enumerate(conteo.values):
    ax[0].text(i, v + 0.2, str(v), ha="center")

corr = df.corr()
im = ax[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(corr.columns)))
ax[1].set_yticks(range(len(corr.columns)))
ax[1].set_xticklabels(corr.columns, rotation=45, ha="right")
ax[1].set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                   fontsize=8)
ax[1].set_title("Matriz de correlación")
fig.colorbar(im, ax=ax[1], fraction=0.046)
plt.tight_layout()
plt.savefig("fig1_exploracion.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CELDA 3. VARIABLES X e y Y DIVISIÓN TRAIN / TEST
# ------------------------------------------------------------
variables = ["edad", "salario", "años_empresa", "satisfaccion", "horas_extra"]
X = df[variables]
y = df["renuncia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print("\n" + "=" * 70)
print("2. DIVISIÓN DE DATOS")
print("=" * 70)
print("Entrenamiento:", X_train.shape, "| Prueba:", X_test.shape)
print("Clases en entrenamiento:", y_train.value_counts().to_dict())
print("Clases en prueba:", y_test.value_counts().to_dict())

# ------------------------------------------------------------
# CELDA 4. REGRESIÓN LOGÍSTICA (ejercicio principal)
# Se escalan las variables porque tienen magnitudes muy distintas
# (salario ~ miles vs. satisfacción ~ 1-10).
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. REGRESIÓN LOGÍSTICA")
print("=" * 70)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

reglog = LogisticRegression(max_iter=1000, random_state=42)
reglog.fit(X_train_sc, y_train)

y_pred_rl = reglog.predict(X_test_sc)
y_prob_rl = reglog.predict_proba(X_test_sc)[:, 1]

print("\nCoeficientes (variables estandarizadas):")
coef = pd.DataFrame({"variable": variables,
                     "coeficiente": reglog.coef_[0]}).sort_values(
    "coeficiente", key=abs, ascending=False)
print(coef.round(4).to_string(index=False))
print("Intercepto:", round(reglog.intercept_[0], 4))

print("\nProbabilidad de renuncia en el conjunto de prueba:")
tabla_prob = X_test.copy()
tabla_prob["real"] = y_test.values
tabla_prob["prob_renuncia"] = y_prob_rl.round(4)
tabla_prob["prediccion"] = y_pred_rl
print(tabla_prob)

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_rl))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_rl, zero_division=0,
                            target_names=["Permanece", "Renuncia"]))

# Figura 2: curva sigmoide sobre la combinación lineal z
z_all = reglog.decision_function(scaler.transform(X))
zz = np.linspace(-6, 6, 200)
sig = 1 / (1 + np.exp(-zz))
plt.figure(figsize=(7, 4.5))
plt.plot(zz, sig, color="#1565c0", label="Función sigmoide")
plt.axhline(0.5, color="gray", ls="--", label="Umbral 0.5")
plt.scatter(z_all[y == 0], np.zeros((y == 0).sum()), c="#2e7d32",
            label="Permanece", edgecolor="k")
plt.scatter(z_all[y == 1], np.ones((y == 1).sum()), c="#c62828",
            label="Renuncia", edgecolor="k")
plt.xlabel("z = b0 + b1x1 + ... + bnxn")
plt.ylabel("Probabilidad de renuncia")
plt.title("Regresión Logística: función sigmoide")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_sigmoide.png", dpi=150)
plt.close()

# Predicción de un nuevo empleado
nuevo = pd.DataFrame({"edad": [29], "salario": [2300], "años_empresa": [2],
                      "satisfaccion": [3], "horas_extra": [12]})
p_nuevo = reglog.predict_proba(scaler.transform(nuevo))[0, 1]
print("\nNuevo empleado:", nuevo.iloc[0].to_dict())
print(f"Probabilidad de renuncia: {p_nuevo:.2%}")
print("Predicción:", "RENUNCIA" if p_nuevo >= 0.5 else "PERMANECE")

# ------------------------------------------------------------
# CELDA 5. MODELOS DE CLASIFICACIÓN
# KNN, SVM y Regresión Logística usan Pipeline con StandardScaler
# (son sensibles a la escala). Árboles y Naive Bayes no lo requieren.
# ------------------------------------------------------------
modelos = {
    "Regresión Logística": Pipeline([("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=1000))]),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": Pipeline([("sc", StandardScaler()),
                     ("m", SVC(kernel="linear", random_state=42))]),
    "KNN": Pipeline([("sc", StandardScaler()),
                     ("m", KNeighborsClassifier(n_neighbors=3))]),
    "Naive Bayes": GaussianNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def puntaje_positivo(modelo, datos):
    """Probabilidad de la clase 1; para SVM (sin predict_proba) se usa la
    función de decisión pasada por una sigmoide (solo como puntaje de ranking)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(datos)[:, 1]
    return 1 / (1 + np.exp(-modelo.decision_function(datos)))

resultados = []
matrices = {}
probs = {}

print("\n" + "=" * 70)
print("4. EVALUACIÓN DE LOS MODELOS DE CLASIFICACIÓN")
print("=" * 70)

for nombre, modelo in modelos.items():
    print("\n" + "-" * 70)
    print("MODELO:", nombre)
    print("-" * 70)

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = puntaje_positivo(modelo, X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc_train = accuracy_score(y_train, modelo.predict(X_train))

    cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    matrices[nombre] = cm
    probs[nombre] = y_prob

    print("Matriz de confusión [[TN FP] [FN TP]]:")
    print(cm)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0,
                                target_names=["Permanece", "Renuncia"]))
    print("Accuracy entrenamiento:", round(acc_train, 4))
    print("Accuracy prueba       :", round(acc, 4))
    print("Validación cruzada (5 pliegues):", cv_scores.round(3))
    print("Promedio CV:", round(cv_scores.mean(), 4),
          "| Desv. estándar:", round(cv_scores.std(), 4))

    resultados.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "CV promedio": cv_scores.mean(),
        "CV desv.": cv_scores.std(),
        "Acc. train": acc_train,
    })

# ------------------------------------------------------------
# CELDA 6. TABLA COMPARATIVA Y MEJOR MODELO
# ------------------------------------------------------------
df_res = pd.DataFrame(resultados).sort_values(
    by=["F1-score", "CV promedio"], ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("5. TABLA COMPARATIVA")
print("=" * 70)
print(df_res.round(4).to_string(index=False))
df_res.round(4).to_csv("resultados_modelos.csv", index=False)

mejor = df_res.iloc[0]
print("\nMejor modelo según F1-score (desempate por validación cruzada):",
      mejor["Modelo"])

# ------------------------------------------------------------
# CELDA 7. GRÁFICAS DE RESULTADOS
# ------------------------------------------------------------
# Figura 3: matrices de confusión de los 6 modelos
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (nombre, cm) in zip(axes.ravel(), matrices.items()):
    disp = ConfusionMatrixDisplay(cm, display_labels=["Permanece", "Renuncia"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(nombre)
plt.suptitle("Matrices de confusión (conjunto de prueba)", fontsize=14)
plt.tight_layout()
plt.savefig("fig3_matrices_confusion.png", dpi=150)
plt.close()

# Figura 4: comparación de métricas
metricas = ["Accuracy", "Precision", "Recall", "F1-score", "CV promedio"]
x = np.arange(len(df_res))
w = 0.16
plt.figure(figsize=(12, 5.5))
for i, m in enumerate(metricas):
    plt.bar(x + i * w, df_res[m], w, label=m)
plt.xticks(x + 2 * w, df_res["Modelo"], rotation=15)
plt.ylim(0, 1.12)
plt.ylabel("Valor de la métrica")
plt.title("Comparación de métricas por modelo")
plt.legend(ncol=5, loc="upper center")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig4_comparacion_metricas.png", dpi=150)
plt.close()

# Figura 5: árbol de decisión (entrenado con todos los datos para visualizar)
arbol = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
plt.figure(figsize=(12, 6))
plot_tree(arbol, feature_names=variables,
          class_names=["Permanece", "Renuncia"], filled=True, rounded=True,
          fontsize=9)
plt.title("Árbol de Decisión (profundidad máxima = 3)")
plt.tight_layout()
plt.savefig("fig5_arbol_decision.png", dpi=150)
plt.close()

# Figura 6: importancia de variables (Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=variables).sort_values()
plt.figure(figsize=(7, 4.5))
plt.barh(imp.index, imp.values, color="#6a1b9a")
plt.xlabel("Importancia")
plt.title("Importancia de variables - Random Forest")
plt.tight_layout()
plt.savefig("fig6_importancia_rf.png", dpi=150)
plt.close()
print("\nImportancia de variables (Random Forest):")
print(imp.sort_values(ascending=False).round(4))

# Figura 7: curvas ROC
plt.figure(figsize=(7, 5.5))
for nombre, p in probs.items():
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_test, p):.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Azar")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos (Recall)")
plt.title("Curvas ROC")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig7_roc.png", dpi=150)
plt.close()

# Figura 8: sobreajuste (accuracy entrenamiento vs. validación cruzada)
plt.figure(figsize=(10, 4.8))
x = np.arange(len(df_res))
plt.bar(x - 0.2, df_res["Acc. train"], 0.4, label="Entrenamiento")
plt.bar(x + 0.2, df_res["CV promedio"], 0.4, label="Validación cruzada")
plt.xticks(x, df_res["Modelo"], rotation=15)
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Entrenamiento vs. validación cruzada (detección de overfitting)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig8_overfitting.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CELDA 8. PREDICCIÓN DEL NUEVO EMPLEADO CON TODOS LOS MODELOS
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("6. PREDICCIÓN PARA UN NUEVO EMPLEADO")
print("=" * 70)
print(nuevo.to_string(index=False))
filas = []
for nombre, modelo in modelos.items():
    modelo.fit(X, y)  # modelo final con todos los datos
    pr = puntaje_positivo(modelo, nuevo)[0]
    filas.append({"Modelo": nombre,
                  "Puntaje/Prob. renuncia": round(pr, 4),
                  "Predicción": "Renuncia" if modelo.predict(nuevo)[0] == 1
                  else "Permanece"})
print(pd.DataFrame(filas).to_string(index=False))

print("\nArchivos generados: fig1..fig8 (.png) y resultados_modelos.csv")
