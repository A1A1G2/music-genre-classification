import pandas as pd

# Verilerin bulunduğu CSV dosyasini okuma
data = pd.read_csv('/home/aag/Desktop/dsp_proje/blackman/test_result.csv', delimiter=';')

# Sinif dağilimi
class_distribution = data['expected_label'].value_counts()

# Toplam doğru tahmin sayisi
correct_predictions = data[data['predicted_label'] == data['expected_label']]
total_correct = len(correct_predictions)

# Toplam yanlış tahmin sayisi
total_incorrect = len(data) - total_correct

# Toplam veri sayisi
total_data = len(data)

# Doğru tahmin yüzdesi
accuracy_percentage = (total_correct / total_data) * 100

# Yanlış tahmin yüzdesi
error_percentage = (total_incorrect / total_data) * 100

# Sinif bazinda doğru tahmin yüzdesi
class_accuracy = correct_predictions.groupby('expected_label').size() / class_distribution * 100

# Sinif bazinda yanlış tahmin yüzdesi
class_error = (class_distribution - correct_predictions.groupby('expected_label').size()) / class_distribution * 100

# İstatistikleri yazdirma
print("Toplam Doğru Tahmin Sayisi:", total_correct)
print("Toplam Yanlış Tahmin Sayisi:", total_incorrect)
print("Toplam Veri Sayisi:", total_data)
print("Doğru Tahmin Yüzdesi: %.2f%%" % accuracy_percentage)
print("Yanlış Tahmin Yüzdesi: %.2f%%" % error_percentage)
print()
print("Sinif Bazinda Doğru Tahmin Yüzdesi:")
print(class_accuracy.to_string())
print()
print("Sinif Bazinda Yanlış Tahmin Yüzdesi:")
print(class_error.to_string())