"""Evaluación complementaria: ambos sistemas a nivel de pareja (22 x 22).
Lee archivos de resultados y rasgos; no modifica salidas originales.
La evaluación de MVKD usa puntuaciones PRECALCULADAS en R, por lo que no
establece ausencia de fuga en la generación de sus puntuaciones de origen.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.isotonic import IsotonicRegression

BASE = Path(__file__).resolve().parents[2]  # repositorio/codigo/python/script.py
OUT = BASE / 'resultados' / 'comparacion_B'
OUT.mkdir(exist_ok=True)
VOCALES = list('aeiou')
CONDS = {
 'limpio_F1234': (['F1_limpio','F2_limpio','F3_limpio','F4_limpio'],['F1_limpio','F2_limpio','F3_limpio','F4_limpio']),
 'limpio_F123': (['F1_limpio','F2_limpio','F3_limpio'],['F1_limpio','F2_limpio','F3_limpio']),
 'telefono_F123': (['F1_tel','F2_tel','F3_tel'],['F1_tel','F2_tel','F3_tel']),
 'cruzado_F123': (['F1_limpio','F2_limpio','F3_limpio'],['F1_tel','F2_tel','F3_tel']),
}

def cllr(s,y):
 return .5*(np.logaddexp(0,-s[y==1]).mean()+np.logaddexp(0,s[y==0]).mean())/np.log(2)
def cmin(s,y):
 weights=np.where(y==1,1/sum(y==1),1/sum(y==0))
 p=IsotonicRegression(out_of_bounds='clip').fit_transform(s,y,sample_weight=weights)
 p=np.clip(p,1e-6,1-1e-6)
 return cllr(np.log(p/(1-p)),y)
def eer(s,y):
 fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; i=np.argmin(abs(fpr-fnr))
 return (fpr[i]+fnr[i])/2

def loso_score(X,y,sus,off,hab):
 acc=[[] for _ in y]
 counts=[]
 for speaker in hab:
  train=(sus!=speaker)&(off!=speaker)
  test=~train
  if not (set(y[train]) == {0,1}): raise ValueError('Fold train one class')
  z=StandardScaler().fit(X[train])
  clf=LogisticRegression(max_iter=5000,class_weight='balanced',random_state=42)
  clf.fit(z.transform(X[train]),y[train])
  for index,score in zip(np.where(test)[0],clf.decision_function(z.transform(X[test]))):
   acc[index].append(float(score))
  counts.append((speaker,int(train.sum()),int(test.sum())))
 if not all(len(x)== (1 if sus[i]==off[i] else 2) for i,x in enumerate(acc)):
  raise ValueError('LOSO not exactly one score for same and two for different')
 return np.array([np.mean(x) for x in acc]),counts

raw=pd.read_csv(BASE/'rasgos'/'formantes.csv')
hab=pd.read_csv(BASE/'resultados'/'mvkd_hablantes.csv')['hablante'].astype(str).tolist()
m=pd.read_csv(BASE/'resultados'/'mvkd_pares.csv')
assert len(hab)==22 and len(set(hab))==22
assert len(m)==4*5*22*22
agg=raw.groupby(['hablante','sesion','vocal']).mean(numeric_only=True)
records=[];detail=[]
for cond,(a,b) in CONDS.items():
 sub=m[m.condicion==cond]
 wide=sub.pivot_table(index=['sus','off','mismo'],columns='vocal',values='logLR').reset_index()
 assert wide.shape[0]==484
 wide=wide.sort_values(['sus','off']).reset_index(drop=True)
 assert wide[VOCALES].notna().all().all()
 ids=list(zip(wide.sus,wide.off))
 assert len(ids)==len(set(ids))==484
 assert set(ids)=={(i,j) for i in hab for j in hab}
 sus=wide['sus'].to_numpy();off=wide['off'].to_numpy();y=wide['mismo'].to_numpy(dtype=int)
 assert np.array_equal(y,(sus==off).astype(int)) and y.sum()==22
 # five vowel-specific acoustic differences; same per-pair index as MVKD
 X=[]
 for i,j in ids:
  row=[]
  for v in VOCALES:
   va=agg.loc[(i,'A',v),a].to_numpy(dtype=float)
   vb=agg.loc[(j,'B',v),b].to_numpy(dtype=float)
   row.extend(np.abs(va-vb).tolist())
  X.append(row)
 X=np.array(X,dtype=float)
 assert np.isfinite(X).all(), f'Non finite feature in {cond}'
 vm=np.clip(wide[VOCALES].to_numpy(dtype=float),-30,30)
 assert np.isfinite(vm).all()
 for system,feats in [('MVKD, fusion LOSO (puntuaciones R originales)',vm),
                       ('Regresion logistica, cinco vocales por pareja',X)]:
  scores,fold_counts=loso_score(feats,y,sus,off,hab)
  met={'condicion':cond,'sistema':system,'hablantes':len(hab),'pares':len(y),
       'mismo':int(sum(y)),'distinto':int(sum(y==0)),
       'Cllr':round(cllr(scores,y),3),'Cllr_min':round(cmin(scores,y),3),
       'EER':round(eer(scores,y),4)}
  records.append(met)
  for (i,j),label,score in zip(ids,y,scores):
   detail.append({'condicion':cond,'sistema':system,'sus':i,'off':j,'mismo':label,'score_logLR':score})
 print(cond,'evaluado: 484 pares para cada sistema')
res=pd.DataFrame(records)
res.to_csv(OUT/'comparacion_B_metricas.csv',index=False)
pd.DataFrame(detail).to_csv(OUT/'comparacion_B_puntuaciones.csv',index=False)
print('\n'+res.to_string(index=False))

# Figuras adicionales para el informe (misma tabla, no se introducen nuevos datos)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
names = ['Limpio F1–F4','Limpio F1–F3','Telefónico F1–F3','Cruzado F1–F3']
conds = list(CONDS.keys())
a = res[res.sistema.str.startswith('MVKD')].set_index('condicion').loc[conds]
b = res[res.sistema.str.startswith('Regresion')].set_index('condicion').loc[conds]
x = np.arange(4); w = .37
for metric, suffix, ylabel, upper in [('Cllr_min','cllr_min','Cllr_min (menor = mejor)',1.03),('EER','eer','EER (%) (menor = mejor)',45)]:
 factor = 100 if metric == 'EER' else 1
 fig,ax=plt.subplots(figsize=(10,5))
 ba=ax.bar(x-w/2,a[metric]*factor,w,label='MVKD, fusión de vocales')
 bb=ax.bar(x+w/2,b[metric]*factor,w,label='Regresión logística, 5 vocales por pareja')
 fmt='%.1f' if metric == 'EER' else '%.3f'
 ax.bar_label(ba,fmt=fmt,padding=3,fontsize=9)
 ax.bar_label(bb,fmt=fmt,padding=3,fontsize=9)
 ax.set_xticks(x,names);ax.set_ylim(0,upper);ax.set_ylabel(ylabel)
 ax.set_title('Comparación homogénea exploratoria (22 hablantes, 484 parejas)')
 ax.legend(loc='upper left',fontsize=9);ax.spines[['top','right']].set_visible(False)
 fig.tight_layout();fig.savefig(OUT/f'figura_B_{suffix}.png',dpi=200,bbox_inches='tight')
 plt.close(fig)
print('Tablas y figuras guardadas en', OUT)
