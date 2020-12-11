import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-colorblind')

#ver os estilos disponíveis, para escolher o seaborn-colorblind que é bonitinho
plt.style.available


df = pd.read_csv('train.csv')
df.shape
df.columns

enem = df[['SG_UF_RESIDENCIA','NU_IDADE','TP_SEXO','TP_COR_RACA','NU_NOTA_MT','NU_NOTA_CH','Q025',
         'Q006','Q002']]
print('meu dataset tem',enem.shape[0],'colunas e',enem.shape[1],'linhas')

enem.head()

pd.DataFrame({'tipos':enem.dtypes, 'missing':enem.isna().sum()})

enem['TP_COR_RACA'] = enem['TP_COR_RACA'].map({0:'NA', 6:'NA', 1:'Branca', 2:'Preta', 3:'Parda',
                                               4:'Amarela', 5:'Indígena'})

enem['Q025'] = enem['Q025'].map({'A':'Não', 'B':'Sim'})

enem['Q002'] = enem['Q002'].map({'A':'Nunca estudou', 'B':'Não completou EF', 'C':'Não completou EF',
                                'D':'Não completou EM', 'E':'Não completou ES', 'F':'Graduada',
                                'G':'Pós-graduada','H':'Não sabe'})

enem['Q006'] = enem['Q006'].map({'A':'Nenhuma renda', 'B':'Até 1.320', 'C':'Até 1.320',
                                 'D':'Até 3.520', 'E':'Até 3.520', 'F':'Até 3.520', 'G':'Até 3.520',
                                 'H':'Até 10.560','I':'Até 10.560', 'J':'Até 10.560','K':'Até 10.560',
                                 'L':'Até 10.560','M':'Até 10.560','N':'Até 10.560',
                                 'O':'Mais de 10.560','P':'Mais de 10.560','Q':'Mais de 10.560'})


#?????????????????????????????????????????????
enem['TP_SEXO'].value_counts()/enem.shape[0]*100

#?????????????????????????????????????????????
enem['TP_COR_RACA'].value_counts()/enem.shape[0]*100


sns.countplot(enem['TP_COR_RACA'])
plt.xlabel('Cor/Raça')
plt.ylabel("")
plt.show()

sns.countplot(enem['TP_SEXO'],hue=enem['TP_COR_RACA'], dodge=True)
plt.title('Sexo x Cor')
plt.xlabel('')
plt.ylabel("")
plt.legend(loc='best')
plt.show()

sns.countplot(enem['SG_UF_RESIDENCIA'], orient='h')
plt.title('UF de Residência', size=15)
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("")
plt.show()


enem['Q025'].value_counts()/enem.shape[0]*100

enem['Q002'].value_counts()/enem.shape[0]*100

sns.countplot(enem['Q002'])
plt.title('Escolaridade da mãe')
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("")
plt.show()

enem['Q006'].value_counts()/enem.shape[0]*100

sns.countplot(enem['Q006'])
plt.title('Renda familiar mensal')
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("")
plt.show()

plt.hist(enem['NU_NOTA_MT'], alpha=0.7, color='red')
plt.hist(enem['NU_NOTA_CH'], alpha=0.7, color='blue')
plt.title('Histograma das notas')
plt.show()

plt.hist(enem['NU_IDADE'],bins=15, histtype='bar', color='#3CB371')
plt.title('Histograma da idade')
plt.show()

plt.scatter(enem['NU_IDADE'], enem['NU_NOTA_MT'], alpha=0.5)
plt.title("Idade x Nota na prova de matemática", size=15)
plt.xlabel("Idade")
plt.ylabel("Nota de matemática")
plt.ylim(200) #ignorando as notas = 0
plt.show()

plt.scatter(enem['NU_IDADE'], enem['NU_NOTA_CH'], alpha=0.5)
plt.title("Idade x Nota na prova de humanas", size=15)
plt.xlabel("Idade")
plt.ylabel("Nota de matemática")
plt.ylim(200) #ignorando as notas = 0
plt.show()

plt.scatter(enem['NU_NOTA_MT'], enem['NU_NOTA_CH'], alpha=0.5)
plt.xlim(300) #removendo as notas =0
plt.ylim(300) #removendo as notas =0
plt.xlabel("Nota de Matemática")
plt.ylabel("Nota de Ciências Humanas")
plt.show()

sns.boxplot(x=enem['TP_SEXO'], y=enem['NU_NOTA_MT'])
plt.xlabel("")
plt.ylabel("Nota de Matemática")
plt.show()

from scipy import stats
from statsmodels.stats import diagnostic

sexo = enem[['TP_SEXO', 'NU_NOTA_MT']]
sexo_f = sexo.query('TP_SEXO == "F"').drop('TP_SEXO',axis=1).dropna()
sexo_m = sexo.query('TP_SEXO == "M"').drop('TP_SEXO',axis=1).dropna()
print(sexo_f.shape[0])
print(sexo_m.shape[0])

print('sexo_f:',diagnostic.lilliefors(sexo_f))
print('sexo_m:',diagnostic.lilliefors(sexo_m))

stats.mannwhitneyu(sexo_f, sexo_m, alternative='two-sided')

sns.boxplot(x=enem['Q025'], y=enem['NU_NOTA_MT'])
plt.xlabel("Tem internet em casa?")
plt.ylabel("Nota de Matemática")
plt.show()

internet = enem[['Q025', 'NU_NOTA_MT']]
internet_n = internet.query('Q025 == "Não"').drop('Q025',axis=1).dropna()
internet_s = internet.query('Q025 == "Sim"').drop('Q025',axis=1).dropna()
print(internet_n.shape[0])
print(internet_s.shape[0])

stats.mannwhitneyu(internet_n, internet_s, alternative='two-sided')

sns.boxplot(x=enem['TP_COR_RACA'], y=enem['NU_NOTA_MT'])
plt.xlabel("")
plt.ylabel("Nota de Matemática")
plt.show()

raca = enem[['TP_COR_RACA', 'NU_NOTA_MT']]
raca_b = raca.query('TP_COR_RACA == "Branca"').drop('TP_COR_RACA',axis=1).dropna()
raca_pa = raca.query('TP_COR_RACA == "Parda"').drop('TP_COR_RACA',axis=1).dropna()
raca_pr = raca.query('TP_COR_RACA == "Preta"').drop('TP_COR_RACA',axis=1).dropna()
raca_a = raca.query('TP_COR_RACA == "Amarela"').drop('TP_COR_RACA',axis=1).dropna()
raca_i = raca.query('TP_COR_RACA == "Indígena"').drop('TP_COR_RACA',axis=1).dropna()

print(raca_b.shape[0])
print(raca_pa.shape[0])
print(raca_pr.shape[0])
print(raca_a.shape[0])
print(raca_i.shape[0])

stats.kruskal(raca_b,raca_pa,raca_pr,raca_a,raca_i)


sns.boxplot(x=enem['Q006'], y=enem['NU_NOTA_MT'])
plt.xticks(rotation=90)
plt.xlabel("Renda familiar")
plt.ylabel("Nota de Matemática")
plt.show()


sns.boxplot(x=enem['Q002'], y=enem['NU_NOTA_MT'])
plt.xticks(rotation=90)
plt.xlabel("Escolaridade da mãe")
plt.ylabel("Nota de Matemática")
plt.show()

sns.boxplot(x=enem['SG_UF_RESIDENCIA'], y=enem['NU_NOTA_MT'])
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("Nota de Matemática")
plt.show()