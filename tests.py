from matplotlib import pyplot as plt

wyniki = [72.67, 76.15, 76.68, 77.53, 77.90, 78.25]
procenty = ["1%", "5%", "25%", "50%", "75%", "100%"]

# plot

plt.bar(procenty, wyniki)
plt.xlabel('Procent danych treningowych')
plt.ylabel('Dokładność [%]')
plt.title('Dokładność modelu w zależności od ilości danych treningowych z akcelerometru')
plt.xticks(rotation=45)
plt.ylim(70, 80)

for i, v in enumerate(wyniki):
    plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
plt.show()
