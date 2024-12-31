import os
print(os.getcwd())
# Klasör yolunu belirtin
folder_path = './'  # Kendi klasör yolunuzu buraya yazın

# Etiket eşleştirmesi
label_mapping = {
    '0':'2',
    '2':'4',
    '3':'0',
    '4':'3'   # person -> 0
}

# Klasördeki tüm .txt dosyalarını işleme
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            print("aaaa")
            if parts:  # Boş satır kontrolü
                label = parts[0]
                if label in label_mapping:
                    parts[0] = label_mapping[label]
                updated_lines.append(' '.join(parts))
        
        with open(file_path, 'w') as file:
            file.write('\n'.join(updated_lines))

print("Etiket güncelleme tamamlandı.")
