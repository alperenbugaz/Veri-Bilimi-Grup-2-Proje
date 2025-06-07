import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math
from scipy.stats import skew, kurtosis
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score,
                             recall_score, f1_score, roc_curve, auc, precision_recall_curve)

# Yeni modeller için importlar
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# --- Görüntü İşleme Kısmı ---

def filter_non_green_lab(image_path, output_path):
    img = cv2.imread(image_path)
    median = cv2.medianBlur(img, 5)
    lab_img = cv2.cvtColor(median, cv2.COLOR_BGR2LAB)
    _, a_channel, _ = cv2.split(lab_img)
    green_mask_lab = cv2.inRange(a_channel, 128, 255)
    result_non_green_lab = cv2.bitwise_and(img, img, mask=green_mask_lab)
    cv2.imwrite(output_path, result_non_green_lab)


def process_images_in_subfolder_filter(subfolder_path, base_output_folder):
    image_files = [
        f.path
        for f in os.scandir(subfolder_path)
        if f.is_file() and f.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    output_folder = os.path.join(base_output_folder, os.path.basename(subfolder_path))
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_files:
        output_file = os.path.join(output_folder, f"processed_{os.path.basename(image_path)}")
        filter_non_green_lab(image_path, output_file)


def run_image_preprocessing(folder_path, output_base_folder="output_preprocessing"):
    print("Görüntü ön işleme adımı başlatılıyor...")
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        process_images_in_subfolder_filter(subfolder, output_base_folder)
    print(f"Görüntü ön işleme tamamlandı. İşlenmiş görüntüler '{output_base_folder}' klasörüne kaydedildi.")
    return output_base_folder


# --- Öznitelik Çıkarımı ve Normalizasyon Kısmı ---


def extract_image_features(image_path, label, data_list):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, thresholded_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    unique_elements, counts_elements = np.unique(thresholded_img, return_counts=True)
    probabilities = counts_elements / np.sum(counts_elements)
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    energy = np.sum(thresholded_img.astype(np.float64) ** 2)  # Olası taşmaları önlemek için float64'e dönüştürme
    moments = cv2.moments(thresholded_img)
    hu_moments = cv2.HuMoments(moments).flatten()
    homogeneity = hu_moments[0] if len(hu_moments) > 0 else 0  # Hu momentleri boş değilse ilkini al

    rmse = sqrt(np.mean((thresholded_img.astype(np.float64) - np.mean(thresholded_img)) ** 2))

    smoothness = 1 - (1 / (1 + np.var(thresholded_img.astype(np.float64))))

    skewness = skew(thresholded_img.flatten())

    kurt = kurtosis(thresholded_img.flatten())

    variance = np.var(thresholded_img.astype(np.float64))

    hist = cv2.calcHist([thresholded_img], [0], None, [256], [0, 256])
    hist_normalized = hist / np.sum(hist)
    contrast_sum = 0
    mean_intensity = np.sum(np.arange(256) * hist_normalized.flatten())  # np.mean değil np.sum olmalı
    for i in range(256):
        contrast_sum += ((i - mean_intensity) ** 2) * hist_normalized[i][0]

    contrast = contrast_sum
    sd = np.std(thresholded_img.astype(np.float64))
    mean_val = np.mean(thresholded_img.astype(np.float64))
    flat_img = thresholded_img.flatten().astype(np.float64)
    correlation = np.correlate(flat_img, flat_img)[0] if flat_img.size > 0 else 0
    inverse_difference_moment = np.sum(1 / (1 + (np.arange(256) - np.arange(256)[:, np.newaxis]) ** 2))

    image_features = [
        os.path.basename(image_path),
        energy,
        homogeneity,
        rmse,
        smoothness,
        skewness,
        kurt,
        contrast,
        sd,
        mean_val,
        entropy_value,
        correlation,
        inverse_difference_moment,
        variance,
        label,
    ]
    image_features = [
        0 if (isinstance(feature, (int, float)) and math.isnan(feature)) else feature
        for feature in image_features
    ]
    data_list.append(image_features)


def process_subfolder_for_features(subfolder_path, data_list):
    image_files = [
        f.path
        for f in os.scandir(subfolder_path)
        if f.is_file() and f.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    label = os.path.basename(subfolder_path)
    for image_path in image_files:
        extract_image_features(image_path, label, data_list)


def feature_extraction(folder_path, output_csv_file="image_features.csv"):
    print("Öznitelik çıkarımı adımı başlatılıyor...")
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    data = []
    for subfolder in subfolders:
        process_subfolder_for_features(subfolder, data)

    column_names = [
        "Image", "Energy", "Homogeneity", "RMSE", "Smoothness", "Skewness", "Kurtosis",
        "Contrast", "SD", "Mean", "Entropy", "Correlation", "Inverse Difference Moment", "Variance", "Label"
    ]
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(output_csv_file, index=False)
    print(f"Öznitelik çıkarımı tamamlandı. Öznitelikler '{output_csv_file}' dosyasına kaydedildi.")
    return output_csv_file


def normalize_features(input_csv_file, output_csv_file="normalized_features.csv"):
    print("Öznitelik normalleştirme adımı başlatılıyor...")
    df = pd.read_csv(input_csv_file)
    if 'Image' in df.columns:
        df = df.drop('Image', axis=1)

    # 'Label' sütunu hariç tüm sayısal sütunları seç
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    df_to_normalize = df[numerical_cols]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_normalized_values = scaler.fit_transform(df_to_normalize)
    df_normalized = pd.DataFrame(df_normalized_values, columns=df_to_normalize.columns, index=df.index)
    # 'Label' sütununu (eğer varsa) geri ekle
    if 'Label' in df.columns:
        df_normalized = pd.concat([df_normalized, df[['Label']]], axis=1)

    df_normalized.to_csv(output_csv_file, index=False)
    print(
        f"Öznitelik normalleştirme tamamlandı. Normalleştirilmiş öznitelikler '{output_csv_file}' dosyasına kaydedildi.")
    return output_csv_file


# --- Sınıflandırma ve Değerlendirme Kısmı ---


def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    print(f"{model_name} için karmaşıklık matrisi oluşturuluyor...")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.xlabel('Tahmin Edilen Etiketler')
    plt.ylabel('Gerçek Etiketler')
    plt.title(f'{model_name} - Karmaşıklık Matrisi')
    plt.show()
    print(f"{model_name} için karmaşıklık matrisi gösterildi.")


def plot_roc_curves(model, X_test, y_test, model_name="Model"):
    print(f"{model_name} için ROC eğrileri oluşturuluyor...")
    plt.figure(figsize=(10, 7))
    unique_classes = np.unique(y_test)
    colors = plt.get_cmap('tab10', len(unique_classes))

    if hasattr(model, "decision_function"):
        y_decision_function = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        y_decision_function = model.predict_proba(X_test)
    else:
        print(
            f"Hata: {model_name} modeli ne 'decision_function' ne de 'predict_proba' metoduna sahip. ROC eğrileri çizilemiyor.")
        return

    for i, class_label in enumerate(unique_classes):
        y_test_binary = (y_test == class_label).astype(int)

        if len(y_decision_function.shape) > 1 and y_decision_function.shape[1] > 1:
            class_idx = np.where(model.classes_ == class_label)[0][0]
            y_scores_class = y_decision_function[:, class_idx]
        else:
            y_scores_class = y_decision_function

        fpr, tpr, _ = roc_curve(y_test_binary, y_scores_class)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors(i), lw=2, label=f'Sınıf {class_label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Rastgele Tahmin', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title(f'{model_name} - Her Sınıf için ROC Eğrileri (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    print(f"{model_name} için ROC eğrileri gösterildi.")


def plot_precision_recall_curves(model, X_test, y_test, model_name="Model"):
    print(f"{model_name} için Precision-Recall eğrileri oluşturuluyor...")
    plt.figure(figsize=(10, 7))
    unique_classes = np.unique(y_test)
    colors = plt.get_cmap('tab10', len(unique_classes))

    if hasattr(model, "predict_proba"):
        y_probas = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        if len(y_scores.shape) == 1:
            y_probas_temp = np.exp(y_scores) / (1 + np.exp(y_scores))
            if len(unique_classes) == 2:
                y_probas = np.vstack((1 - y_probas_temp, y_probas_temp)).T
            else:  # If decision_function returns a single column for multi-class, try to infer or handle
                y_probas = y_scores  # Fallback, might not be true probabilities
        else:
            y_probas = y_scores  # Use scores if multi-column decision_function
    else:
        print(
            f"Hata: {model_name} modeli ne 'predict_proba' ne de 'decision_function' metoduna sahip. Precision-Recall eğrileri çizilemiyor.")
        return

    for i, class_label in enumerate(unique_classes):
        y_test_binary = (y_test == class_label).astype(int)

        if hasattr(model, "predict_proba"):
            class_idx = np.where(model.classes_ == class_label)[0][0]
            probas_class = y_probas[:, class_idx]

        elif hasattr(model, "decision_function"):
            if len(y_scores.shape) > 1 and y_scores.shape[1] > 1:
                class_idx = np.where(model.classes_ == class_label)[0][0]
                probas_class = y_scores[:, class_idx]
            else:
                probas_class = y_scores

        precision, recall, _ = precision_recall_curve(y_test_binary, probas_class)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors(i), lw=2, label=f'Sınıf {class_label} (AUC = {pr_auc:.2f})')

    plt.xlabel('Duyarlılık (Recall)')
    plt.ylabel('Hassasiyet (Precision)')
    plt.title(f'{model_name} - Her Sınıf için Precision-Recall Eğrileri')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    print(f"{model_name} için Precision-Recall eğrileri gösterildi.")


def plot_f1_scores_per_class(y_test, y_pred, model_name="Model"):
    print(f"{model_name} için Sınıf başına F1 skorları grafiği oluşturuluyor...")
    f1_scores_per_class = []
    unique_classes = sorted(np.unique(y_test))

    for class_label in unique_classes:
        y_test_binary = (y_test == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        f1_scores_per_class.append(f1)

    plt.figure(figsize=(10, 6))
    class_labels_str = [str(c) for c in unique_classes]

    cmap = plt.get_cmap('tab10', len(unique_classes))
    plt.bar(class_labels_str, f1_scores_per_class, color=cmap.colors)

    plt.xlabel('Sınıf Etiketi')
    plt.ylabel('F1-Skoru')
    plt.title(f'{model_name} - Her Sınıf için F1-Skorları')
    plt.ylim([0, 1.05])
    plt.grid(axis='y', linestyle='--')
    for i, v in enumerate(f1_scores_per_class):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
    plt.show()
    print(f"{model_name} için Sınıf başına F1 skorları grafiği gösterildi.")


def train_and_evaluate_multiple_models(X, y):
    print("\n--- Birden Fazla Modelin GridSearchCV ile Eğitimi ve Değerlendirmesi Başlatılıyor ---")

    models_and_params = {
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "param_grid": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf']
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "param_grid": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']  # 'liblinear' for L1/L2, 'lbfgs' for L2
            }
        }
    }

    # results = {}
    best_estimators = {}  # To store the best trained model for each type
    best_model = None
    best_name = None
    best_score = 0
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, mp in models_and_params.items():
        print(f"\n--- Model: {name} için GridSearchCV Başlatılıyor ---")
        grid_search = GridSearchCV(mp["model"], mp["param_grid"], cv=kfold, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_

        best_estimators[name] = best_estimator  # Store the best model

        print(f"\n--- En İyi Parametrelerle Modelin Yeniden Eğitilmesi (Eğitim Verisi Üzerinde) ---: {best_params}")
        print(f"En İyi Çapraz Doğrulama Doğruluğu: {best_score:.4f}")

        # Perform cross-validation with the best estimator to get scores for plotting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                            stratify=y)  # stratify eklendi

        best_estimator.fit(X_train, y_train)  # Sadece eğitim verisiyle eğit

        y_pred = best_estimator.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        print("\n--- Test Seti Değerlendirme Sonuçları ---")
        print("Test doğruluğu:", test_accuracy)
        print("Sınıflandırma Raporu:\n",
              classification_report(y_test, y_pred, zero_division=0))  # zero_division eklendi
        print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

        # results[name] = test_accuracy
        if best_score < test_accuracy:
            best_model = best_estimator
            best_name = name
        # ROC-AUC, Precision, Recall, F1-score
        y_pred_proba = best_estimator.predict_proba(X_test)
        roc_auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        print("Ortalama ROC-AUC (One-vs-Rest):", roc_auc_ovr)

        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        print("Ortalama Hassasiyet (Precision - Makro):", precision_macro)
        print("Ortalama Duyarlılık (Recall - Makro):", recall_macro)
        print("Ortalama F1-Skoru (Makro):", f1_macro)
        print(f"{name} modeli eğitimi ve değerlendirmesi tamamlandı.")

    print("\nTüm modellerin eğitimi ve değerlendirmesi tamamlandı.")
    return best_model, best_name, best_estimators


def plot_all_models_f1_scores_bar(best_estimators, X_test, y_test):
    print("\nTüm modeller ve sınıflar için F1 skorları karşılaştırma çubuk grafiği oluşturuluyor...")

    f1_data = []
    unique_classes = sorted(np.unique(y_test))

    for model_name, model in best_estimators.items():
        y_pred = model.predict(X_test)
        for class_label in unique_classes:
            y_test_binary = (y_test == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)
            f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
            f1_data.append({'Model': model_name, 'Class': class_label, 'F1-Score': f1})

    df_f1 = pd.DataFrame(f1_data)

    plt.figure(figsize=(14, 8))
    # Use 'Model' as x-axis, 'F1-Score' as y-axis, and 'Class' for hue (grouping)
    ax = sns.barplot(x='Model', y='F1-Score', hue='Class', data=df_f1, palette='tab10')

    # Add F1-score values on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

    plt.xlabel('Model')
    plt.ylabel('F1-Skoru')
    plt.title('Her Model ve Her Sınıf için F1-Skorları Karşılaştırması')
    plt.ylim([0, 1.1])  # Adjust y-limit to make space for labels
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Sınıf', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside to prevent overlap
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()
    print("Tüm modeller ve sınıflar için F1 skorları karşılaştırma çubuk grafiği gösterildi.")


# --- Ana İşlev ---
def main():
    # 1. Adım: Görüntü Ön İşleme (Yeşil Olmayan Bölgeleri Filtreleme)
    dataset_folder = "dataset"  # Kendi veri seti klasörünüzün adını girin
    preprocessed_output_folder = "output_preprocessed_images"
    run_image_preprocessing(dataset_folder, preprocessed_output_folder)

    # 2. Adım: Öznitelik Çıkarımı
    feature_extraction_input_folder = preprocessed_output_folder
    raw_features_csv = "image_features_raw.csv"
    feature_extraction(feature_extraction_input_folder, raw_features_csv)
    # 3. Adım: Öznitelik Normalleştirme
    normalized_features_csv = "normalized_features.csv"
    normalize_features(raw_features_csv, normalized_features_csv)
    # 4. Adım: Modellerin GridSearchCV ile Eğitimi ve Değerlendirmesi
    df_normalized = pd.read_csv(normalized_features_csv)
    X = df_normalized.drop('Label', axis=1)
    y = df_normalized['Label']

    # Birden fazla modelin Grid Search ile eğitilmesi ve en iyi modellerin alınması
    best_model, model_name, best_estimators = train_and_evaluate_multiple_models(X, y)

    # 5. Adım: Sonuçların Görselleştirilmesi
    # Split data for test set evaluation of the best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Plot all models' F1 scores on a single bar figure
    plot_all_models_f1_scores_bar(best_estimators, X_test, y_test)

    best_model.fit(X_train, y_train)  # Sadece eğitim verisiyle eğit
    y_pred = best_model.predict(X_test)

    plot_confusion_matrix(y_test, y_pred, model_name=model_name)
    plot_roc_curves(best_model, X_test, y_test, model_name=model_name)
    plot_precision_recall_curves(best_model, X_test, y_test, model_name=model_name)
    plot_f1_scores_per_class(y_test, y_pred, model_name=model_name)

    print("\nTüm işlemler tamamlandı.")


if __name__ == "__main__":

    if not os.path.exists("output_preprocessed_images"):
        os.makedirs("output_preprocessed_images")
    main()