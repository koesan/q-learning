import numpy as np

# Öncellikle labirenti oluşturulur
# Labirentit satır ve sutun sayısı
rows = 11 
columns = 11
# -100. labirentin duvarları bu duvarları, -1. labirentin yolları 100. ise labirentin çıkışı olacak
# Kolaylık olsun diye labirentin tüm değerlerini -100. olarak ayarla
rewards = np.full((rows, columns), -100.)
# Labirentin çıkışını 100. olarak ayarla
rewards[3, 10] = 100.
# Labirentin yollarını -1 olarak ayarla
aisles = {
    1: [i for i in range(1, 10)],
    2: [1, 7, 9],
    3: [i for i in range(1, 8)] + [9],
    4: [3, 7],
    5: [i for i in range(1,10)],
    6: [5],
    7: [i for i in range(1, 10)],
    8: [3, 7,5],
    9: [i for i in range(1,10) ]
}
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.
# Labirenti ekrana bas
print("*****************************LABİRENT******************************")
for row in rewards:
    print(row)
print("*******************************************************************")
# artından labirentin boyu ile aynı boyda yani 11x11 ama hareket yönleri ileri geri sağa ve sola olarak ayarlanan 3b bir q tablosu oluştur
# Q tablosunun başlangıç değerleri 0 dır
q_values = np.zeros((rows, columns, 4))
# Q learning için gerekli olan parametreleri tanımla
lr = 0.9
gamma = 0.9
epsilon = 0.9
total_episodes = 1000 # Kaç defa eğitilecek q tablosu

# bu fonkiyon anlık olarak olduğun yerin değerine bakar bu sayede çıkışa vardınmı veya duvara çarpıp çarmadığını kontrol eder
def kontrol(state):
    return rewards[state[0]][state[1]] == -1.

# eğitime başla
for episode in range(total_episodes):
    # Öncelikle rast gele bir başlangıç konumu seç bu konum ne duvara çarpacak nede çıkışa
    while True:
        state = [np.random.randint(rows), np.random.randint(columns)]
        if kontrol(state):
            break

    while kontrol(state):
        old_state = state.copy()
        # Epsilon-greedy yöntemi ile epsilon değerine göre rastgele hareketler seçmesini sağlar.
        if np.random.random() > epsilon:
            action = np.random.randint(4) # 0 ile 3 arasında rastgele sayılar seçer 0 yukarı 1 aşağı 2 sağa 3 sola
        else:
            action = np.argmax(q_values[state[0], state[1]])
        # Konuma göre seçilen action hareketini uygula
        if action == 0 and old_state[0] > 0:
            state[0] -= 1

        elif action == 1 and old_state[1] < 10:
            state[1] += 1

        elif action == 2 and old_state[0] < 10:
            state[0] += 1

        elif action == 3 and old_state[1] > 0:
            state[1] -= 1

        reward = rewards[state[0], state[1]] # gidilen komdaki ödülü al
        old_q_value = q_values[old_state[0], old_state[1], action]  # Önceki konumdaki q tablosunun değerini old_q_value değişkenine alındı
        td = reward + (gamma * np.max(q_values[state[0], state[1]])) - old_q_value # Elde edilen ödül ile bir sonraki durumun en yüksek Q-değeri arasındaki fark.
        new_q_value = old_q_value + (lr * td) # yeni q verisini hesapla
        q_values[old_state[0], old_state[1], action] = new_q_value # yeni q versini q tablosuna ekle

# Bu fonksiyon ile eğitm sonucu oluşan Q tablosunu kullanarak en kısa çıkışı bulbiliriz
def path(state):

    if not kontrol(state):
        return []

    else:
        current_state = state
        shortest_path = []
        shortest_path.append([current_state[0], current_state[1]])

        while kontrol(current_state):
            action = np.argmax(q_values[current_state[0], current_state[1]])

            if action == 0 and current_state[0] > 0:
                current_state[0] -= 1

            elif action == 1 and current_state[1] < 10:
                current_state[1] += 1

            elif action == 2 and current_state[0] < 10:
                current_state[0] += 1

            elif action == 3 and current_state[1] > 0:
                current_state[1] -= 1

            shortest_path.append([current_state[0], current_state[1]])
        return shortest_path

# Başlangıç konumu
state = [9, 1]
shortest_path = path(state)

print(f"En kısa yol = {shortest_path}")
