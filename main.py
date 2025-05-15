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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,roc_auc_score,precision_score,recall_score, f1_score,roc_curve,auc,precision_recall_curve)


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
    print(f"Öznitelik normalleştirme tamamlandı. Normalleştirilmiş öznitelikler '{output_csv_file}' dosyasına kaydedildi.")
    return output_csv_file

# --- Sınıflandırma ve Değerlendirme Kısmı ---

def train_and_evaluate_svm(X, y):

    print("SVM modeli eğitimi ve değerlendirmesi başlatılıyor...")
    C_values = [0.1, 1, 10, 100]
    kernel_types = ['linear', 'poly', 'rbf']
    best_params = None
    best_mean_accuracy = 0

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Çapraz Doğrulama ile En İyi Parametrelerin Bulunması ---")
    for C_value in C_values:
        for kernel_value in kernel_types:
            svm_model = SVC(C=C_value, kernel=kernel_value, decision_function_shape='ovr', probability=True, random_state=42)
            accuracy_scores = cross_val_score(svm_model, X, y, cv=kfold, scoring='accuracy')
            mean_accuracy = np.mean(accuracy_scores)
            print(f"Parametreler: C={C_value}, Kernel={kernel_value}")
            print("Her katman için doğruluk skorları:", accuracy_scores)
            print("Ortalama doğruluk:", mean_accuracy)
            print("-----------------------------")
            if mean_accuracy > best_mean_accuracy:
                best_mean_accuracy = mean_accuracy
                best_params = {'C': C_value, 'kernel': kernel_value}

    print("\nEn İyi Parametreler:", best_params)

    best_svm_model = SVC(C=best_params['C'], kernel=best_params['kernel'], decision_function_shape='ovr', probability=True, random_state=42)
    best_svm_model.fit(X, y) # Tüm veriyle eğitiliyor.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify eklendi

    print("\n--- En İyi Parametrelerle Modelin Yeniden Eğitilmesi (Eğitim Verisi Üzerinde) ---")
    # En iyi parametrelerle modeli sadece eğitim verisi üzerinde tekrar eğit
    final_svm_model = SVC(C=best_params['C'], kernel=best_params['kernel'], decision_function_shape='ovr', probability=True, random_state=42)
    final_svm_model.fit(X_train, y_train) # Sadece eğitim verisiyle eğit

    y_pred = final_svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Test Seti Değerlendirme Sonuçları ---")
    print("Test doğruluğu:", test_accuracy)
    print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred, zero_division=0)) # zero_division eklendi
    print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))

    # ROC-AUC, Precision, Recall, F1-score

    y_pred_proba = final_svm_model.predict_proba(X_test)
    roc_auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    print("Ortalama ROC-AUC (One-vs-Rest):", roc_auc_ovr)

    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print("Ortalama Hassasiyet (Precision - Makro):", precision_macro)
    print("Ortalama Duyarlılık (Recall - Makro):", recall_macro)
    print("Ortalama F1-Skoru (Makro):", f1_macro)
    print("SVM modeli eğitimi ve değerlendirmesi tamamlandı.")

    return final_svm_model, X_test, y_test, y_pred, kfold

def plot_kfold_cross_validation_results(X, y, C_values, kernel_types, kfold):

    print("K-Katlamalı Çapraz Doğrulama sonuçları grafiği oluşturuluyor...")
    accuracy_results = []
    color_dict = {'linear': 'b', 'poly': 'g', 'rbf': 'r', 'sigmoid': 'm'} # sigmoid eklendi, gerekirse

    for C_value in C_values:
        for kernel_value in kernel_types:
            svm_model = SVC(C=C_value, kernel=kernel_value, decision_function_shape='ovr', probability=True, random_state=42)
            accuracy_scores = cross_val_score(svm_model, X, y, cv=kfold, scoring='accuracy')
            mean_accuracy = np.mean(accuracy_scores)
            accuracy_results.append((C_value, kernel_value, mean_accuracy))

    plt.figure(figsize=(10, 7))
    for kernel_value in kernel_types:
        kernel_data = [(r[0], r[2]) for r in accuracy_results if r[1] == kernel_value]
        if kernel_data:
            c_vals, acc_vals = zip(*kernel_data)
            plt.plot(c_vals, acc_vals, marker='o', linestyle='-', label=f'{kernel_value.capitalize()} Kernel', color=color_dict.get(kernel_value, 'k'))

    plt.xscale('log')
    plt.xlabel('C Parametresi (log ölçeği)')
    plt.ylabel('Ortalama Doğruluk')
    plt.title('SVM için K-Katlamalı Çapraz Doğrulama Sonuçları')
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Kernel Türü')
    plt.show()
    print("K-Katlamalı Çapraz Doğrulama sonuçları grafiği gösterildi.")

def plot_confusion_matrix(y_test, y_pred):

    print("Karmaşıklık matrisi oluşturuluyor...")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels = np.unique(y_test),
                yticklabels= np.unique(y_test))
    plt.xlabel('Tahmin Edilen Etiketler')
    plt.ylabel('Gerçek Etiketler')
    plt.title('Karmaşıklık Matrisi')
    plt.show()
    print("Karmaşıklık matrisi gösterildi.")

def plot_roc_curves(model, X_test, y_test):

    print("ROC eğrileri oluşturuluyor...")
    plt.figure(figsize=(10, 7))
    unique_classes = np.unique(y_test)
    colors = plt.get_cmap('tab10', len(unique_classes))  # Her sınıf için farklı renk
    if hasattr(model, "decision_function"):
        y_decision_function = model.decision_function(X_test)
    else:
        print("Uyarı: Modelde 'decision_function' metodu bulunmuyor. ROC eğrileri için olasılıklar kullanılacak.")
        if hasattr(model, "predict_proba"):
            y_decision_function = model.predict_proba(X_test)
        else:
            print("Hata: Model ne 'decision_function' ne de 'predict_proba' metoduna sahip. ROC eğrileri çizilemiyor.")
            return


    for i, class_label in enumerate(unique_classes):
        y_test_binary = (y_test == class_label).astype(int)

        # Eğer model çok sınıflıysa ve decision_function çoklu sütun döndürüyorsa
        if len(y_decision_function.shape) > 1 and y_decision_function.shape[1] > 1:
            # Sınıfın model.classes_ içindeki indeksini bul
            class_idx = np.where(model.classes_ == class_label)[0][0]
            y_scores_class = y_decision_function[:, class_idx]

        # Eğer model ikili veya decision_function tek sütun döndürüyorsa
        else:
            y_scores_class = y_decision_function # Ya da y_decision_function.ravel()

        fpr, tpr, _ = roc_curve(y_test_binary, y_scores_class)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors(i), lw=2, label=f'Sınıf {class_label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Rastgele Tahmin', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title('Her Sınıf için ROC Eğrileri (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    print("ROC eğrileri gösterildi.")

def plot_precision_recall_curves(model, X_test, y_test):

    print("Precision-Recall eğrileri oluşturuluyor...")
    plt.figure(figsize=(10, 7))
    unique_classes = np.unique(y_test)
    colors = plt.get_cmap('tab10', len(unique_classes))

    if hasattr(model, "predict_proba"):
        y_probas = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        if len(y_scores.shape) == 1: # İkili sınıflandırma veya OVR'da tek skor
            y_probas_temp = np.exp(y_scores) / (1 + np.exp(y_scores)) # Basit sigmoid benzeri dönüşüm
            if len(unique_classes) == 2: # İkili ise iki sütunlu yap
                 y_probas = np.vstack((1-y_probas_temp, y_probas_temp)).T


    for i, class_label in enumerate(unique_classes):
        y_test_binary = (y_test == class_label).astype(int)

        if hasattr(model, "predict_proba"):
            class_idx = np.where(model.classes_ == class_label)[0][0]
            probas_class = y_probas[:, class_idx]

        elif hasattr(model, "decision_function"):
            if len(y_scores.shape) > 1 and y_scores.shape[1] > 1: # Çok sınıflı, çok skorlu
                class_idx = np.where(model.classes_ == class_label)[0][0]
                probas_class = y_scores[:, class_idx] # Skorları doğrudan kullan

            else: # Tek skorlu (ikili veya OVR)
                probas_class = y_scores # Skorları doğrudan kullan (pozitif sınıfın skoru)

        precision, recall, _ = precision_recall_curve(y_test_binary, probas_class)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors(i), lw=2, label=f'Sınıf {class_label} (AUC = {pr_auc:.2f})')

    plt.xlabel('Duyarlılık (Recall)')
    plt.ylabel('Hassasiyet (Precision)')
    plt.title('Her Sınıf için Precision-Recall Eğrileri')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    print("Precision-Recall eğrileri gösterildi.")

def plot_f1_scores_per_class(y_test, y_pred):

    print("Sınıf başına F1 skorları grafiği oluşturuluyor...")
    f1_scores_per_class = []
    unique_classes = sorted(np.unique(y_test)) # Sınıfları sıralı alalım

    for class_label in unique_classes:
        y_test_binary = (y_test == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        f1_scores_per_class.append(f1)

    plt.figure(figsize=(10, 6))
    class_labels_str = [str(c) for c in unique_classes] # X ekseni etiketleri için string

    cmap = plt.get_cmap('tab10', len(unique_classes))
    plt.bar(class_labels_str, f1_scores_per_class, color=cmap.colors)

    plt.xlabel('Sınıf Etiketi')
    plt.ylabel('F1-Skoru')
    plt.title('Her Sınıf için F1-Skorları')
    plt.ylim([0, 1.05])
    plt.grid(axis='y', linestyle='--')
    # Her çubuk için F1 skorunu yazdırma
    for i, v in enumerate(f1_scores_per_class):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
    plt.show()
    print("Sınıf başına F1 skorları grafiği gösterildi.")

# --- Ana İşlev ---
def main():

    # 1. Adım: Görüntü Ön İşleme (Yeşil Olmayan Bölgeleri Filtreleme)
    dataset_folder = "dataset" # Kendi veri seti klasörünüzün adını girin
    preprocessed_output_folder = "output_preprocessed_images"
    run_image_preprocessing(dataset_folder, preprocessed_output_folder)


    # 2. Adım: Öznitelik Çıkarımı
    feature_extraction_input_folder = preprocessed_output_folder
    raw_features_csv = "image_features_raw.csv"
    feature_extraction(feature_extraction_input_folder, raw_features_csv)
    # 3. Adım: Öznitelik Normalleştirme
    normalized_features_csv = "normalized_features.csv"
    normalize_features(raw_features_csv, normalized_features_csv)
    # 4. Adım: SVM Modeli Eğitimi ve Değerlendirmesi
    df_normalized = pd.read_csv(normalized_features_csv)
    X = df_normalized.drop('Label', axis=1)
    y = df_normalized['Label']
    best_svm_model, X_test, y_test, y_pred, kfold_obj = train_and_evaluate_svm(X, y)
    # 5. Adım: Sonuçların Görselleştirilmesi
    plot_kfold_cross_validation_results(X, y, [0.1, 1, 10, 100], ['linear', 'poly', 'rbf'], kfold_obj)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curves(best_svm_model, X_test, y_test)
    plot_precision_recall_curves(best_svm_model, X_test, y_test)
    plot_f1_scores_per_class(y_test, y_pred)

    print("\nTüm işlemler tamamlandı.")

if __name__ == "__main__":

    if not os.path.exists("output_preprocessed_images"):
        os.makedirs("output_preprocessed_images")
    main()