"""
Created on Fri Jan 17 19:23:11 2022
@author: nedir ymamov
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

boy = 72 # geminin boyu (LBP)
genislik = boy / 8 # geminin tam genişiliği
draft = genislik / 2.5
derinlik = 1.5 * draft
cb = .7 # blok katsayısı
rho = 1.025 # deniz suyun yoğununluğu
w = boy * genislik * draft * cb * rho # deplasman hesabı
offset = np.loadtxt("s60.txt", dtype = float) # boyutsuz offset tablosu
offset *= genislik / 2

suhatti = np.array([0, .3, 1, 2, 3, 4, 5, 6]) * draft / 4
posta0 = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) * boy / 10
posta = np.linspace(0, boy, 101)

# offset tablosunu genişletilmesi
offset_yeni = np.zeros((101, 8))
for i in range(8):
    f = interp1d(posta0, offset[:, i], kind = "cubic")
    offset_yeni[:, i] = f(posta)
alan = np.zeros((101, 8))
for i in range(101):
    alan[i, 1:] = 2 * cumtrapz(offset_yeni[i], suhatti)

# atalet moment dağılımı
Iy = 3 * .189 * boy / 100 # orta kesit atalet momenti
Ix = np.empty(101)
Ix[:5] = np.linspace(0, Iy / 2, 5)
Ix[5 : 35] = np.linspace(Iy / 2, Iy, 30)
Ix[35 : 76] = Iy
Ix[76 : 96] = np.linspace(Iy, Iy / 4, 20)
Ix[96:] = np.linspace(Iy / 4, 0, 5)

# denize indirme

# çelik tekne ağırlığı dağılımı
N = boy * genislik * derinlik
cs = (.21 - .026 * np.log10(N)) * (1 + .025 * (boy / derinlik - 12))
G = cs * N * (1 + (2 / 3) * (cb - .7))
a = .68 * G / boy
b = 1.185 * G / boy
c = .58 * G / boy
qx = np.zeros(101)
qx[:33] = np.linspace(a, b, 33)
qx[33 : 68] = b
qx[68:] = np.linspace(b, c, 33)

for i in range(35, 101):
    r =  posta[i] * np.tan(2.5 * np.pi / 180)
    batan = np.linspace(r, 0, i)
    y = np.zeros(i)
    for j in range(i):
        y[j] = np.interp(batan[j], suhatti, alan[j, :]) * rho
    mesafe = np.linspace(posta[:i], 0, len(posta[:i]))
    M1 = 0
    for j in range(i - 1):
        M1 +=  posta[j] * y[j]
    M2 = 0
    for j in range(i - 1):
        M2 +=  posta[j] * qx[j]
    cd = abs( (M1 / np.trapz(qx[:i], posta[:i])) - (M2 / np.trapz(qx[:i], posta[:i])) )
    if cd < .7:
        ax = np.zeros(101)
        ax[:i] = y
        break

# sephiye ve ağırlık dağılımının plot çıktısı
plt.figure(figsize = (10, 3))
plt.title("Denize İndirme")
plt.plot( posta, ax,  posta, qx)
plt.fill_between(posta, ax, color = "g", alpha = .5, hatch = "//", edgecolor = "r")
plt.fill_between(posta, qx, color = "r", alpha = .5, hatch = "o", edgecolor = "g")
plt.legend(["ax", "qx"])

px = ax - qx
Qx = np.array([0, *cumtrapz(px,  posta)])
Mx = np.array([0, *cumtrapz(Qx,  posta)])

# gerilme hesabı
ymax = .6 * derinlik
gerilme = -9.81 * Mx[1 : -1] * ymax / (Ix[1 : -1] * 1000)
gerilme = [0, *gerilme, 0]

# Baş papet geminin tam başında 0.posta old. kabul edildi
# Kesme kuvvetin max old. postada max kayma gerilmesi
# Bu postada eninde perde olduğunu kabul edildi
# Perde sanki dikörtgen kesitli gibi hesaplama yapıldı
n = -15 # baş papatten bir kaç potsa önce
A1 = derinlik * genislik
# Tarafsız eksenin 0.4xD old. kabul edildi
A2 = .4 * derinlik * genislik
S = (A1 - A2) * (.6 * derinlik) / 2
kayma = -9.81 * Qx[n] * S / (genislik * Ix[n] * 1000)
print("Denize indirme")
print("Kesme kuvvetin max old. postada max kayma gerilmesi")
print(round(kayma, 3))

# Gerilme dağılımının plot çıktısı
plt.figure(figsize = (10, 3))
plt.title("Denize İndrime")
plt.plot( posta[:-7], gerilme[:-7])
plt.fill_between(posta[:-7], gerilme[:-7], color = "g", alpha = .4)
plt.legend(["gerilme"])


# dipten yaralanma durumu

# toplam gemi ağırlık dağılımı
qx = np.zeros(101)
a = .68 * w / boy
b = 1.187 * w / boy
c = .58 * w / boy
qx[:33] = np.linspace(a, b, 33)
qx[33 : 68] = b
qx[68:] = np.linspace(b, c, 33)

Ix -= .15 * Ix

ax = alan[:, 7] * rho
ax[36:64] = np.zeros(28)

# sephiye ve ağırlık dağılımının plot çıktısı
plt.figure(figsize = (10, 3))
plt.title("Dipten Yaralanma")
plt.plot( posta, ax,  posta, qx)
plt.fill_between(posta, ax, color = "g", alpha = .5, hatch = "//", edgecolor = "r")
plt.fill_between(posta, qx, color = "r", alpha = .5, hatch = "o", edgecolor = "g")
plt.legend(["ax", "qx"])

px = ax - qx
Qx = np.zeros(101)
Qx[1:] = cumtrapz(px,  posta)

# lineer düzenleme 3%max(Q)
lineer = np.linspace(0, Qx[-1], 101)
Qx -= lineer
Mx = np.zeros(101)
Mx[1:] = cumtrapz(Qx,  posta)

# lineer düzenleme 6%max(M)
lineer = np.linspace(0, Mx[-1], 101)
Mx -= lineer

# gerilme hesabı
gerilme = 9.81 * Mx[1 : -1] * ymax / (Ix[1 : -1] * 1000)
gerilme = [0, *gerilme, 0]

# Kesme kuvvetin max old. postada max kayma gerilmesi
# Bu postada eninde perde olduğunu kabul edildi
# Perde sanki dikörtgen kesitli gibi hesaplama yapıldı
A1 = derinlik * genislik
# Tarafsız eksenin 0.4xD old. kabul edildi
A2 = .4 * derinlik * genislik
S = (A1 - A2) * (.6 * derinlik) / 2
kayma = -9.81 * Qx[n] * S / (genislik * Ix[n] * 1000)
print("\nDipten yaralanma")
print("Kesme kuvvetin max old. postada max kayma gerilmesi")
print(round(kayma, 3))

# gerilme dağılımının plot çıktısı
plt.figure(figsize = (10, 3))
plt.title("Dipten Yaralanma")
plt.plot(posta, gerilme)
plt.legend(["gerilme"])
plt.fill_between(posta, gerilme, color = "g", alpha = .4)