"""MVKD con recalculo integro de estadisticas de fondo por comparacion.

Reimplementacion numerica en Python de las expresiones de fvclrr::MVKD/multivar_kernel_LR.
Se validaron 9.239/9.240 comparaciones de hablantes distintos respecto a la
version R original con diferencias <=1.1e-9; un caso extremo se satura en R.

Diferencias con MVKD original: fondo excluyente recalculado desde cero (incluida
media global); para parejas del mismo hablante se promedia el logLR resultante
de las 21 elecciones posibles del segundo hablante excluido. Evita aleatoriedad.
No afirma ser el propio paquete R; se trata de evaluacion de sensibilidad.

Entrada: rasgos/formantes.csv y resultados/mvkd_hablantes.csv
Salida NUEVA: resultados/mvkd_pares_fondo_corregido_python.csv
No sobrescribe los resultados originales.
"""
import numpy as np, pandas as pd, time
from scipy.special import logsumexp
from scipy.linalg import solve
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
raw=pd.read_csv(BASE/'rasgos'/'formantes.csv')
hab=pd.read_csv(BASE/'resultados'/'mvkd_hablantes.csv').hablante.tolist()
COND={'limpio_F1234':(['F1_limpio','F2_limpio','F3_limpio','F4_limpio'],['F1_limpio','F2_limpio','F3_limpio','F4_limpio']), 'limpio_F123':(['F1_limpio','F2_limpio','F3_limpio'],)*2,'telefono_F123':(['F1_tel','F2_tel','F3_tel'],)*2,'cruzado_F123':(['F1_limpio','F2_limpio','F3_limpio'],['F1_tel','F2_tel','F3_tel'])}

def score(S,O,bg,sus,off,removed_extra=None):
    # background per speaker originally from session A for each condition/vowel
    kept = [h for h in hab if h not in (sus,off)]
    if sus==off: kept.remove(removed_extra if removed_extra is not None else next(h for h in kept))
    groups=[bg[h] for h in kept]
    ni=np.array([len(a) for a in groups]); n=ni.sum(); k=len(groups); d=S.shape[1]
    means=np.array([a.mean(axis=0) for a in groups]); mu=(means*ni[:,None]).sum(axis=0)/n
    within=np.sum([len(a)*((a-a.mean(0)).T@(a-a.mean(0))) for a in groups],axis=0)
    within=(within*k/n)/(n-k)
    between=np.sum([len(a)*np.outer(a.mean(0)-mu,a.mean(0)-mu) for a in groups],axis=0)
    kernel=(4/(k*(2*d+1)))**(2/(d+4))*(between/(k-1)-within)*k/n
    sw=solve(within,np.eye(d),assume_a='sym')
    ks=solve(kernel,np.eye(d))
    ns=len(S);no=len(O);sm=S.mean(0);om=O.mean(0)
    sinv=sw*ns;oinv=sw*no
    typ=solve(oinv+sinv, oinv@om + sinv@sm)
    offk=solve(within/no+kernel,np.eye(d))
    susk=solve(within/ns+kernel,np.eye(d))
    off_sus_inv=sw*ns*no/(ns+no)
    bothk=solve(within/(ns+no)+kernel,np.eye(d))
    def logsumquad(X,inv):
        D=means-X
        return logsumexp(-.5*np.einsum('ij,jk,ik->i',D,inv,D))
    result=.5*(np.linalg.slogdet(kernel)[1]+np.linalg.slogdet(oinv+ks)[1]+np.linalg.slogdet(sinv+ks)[1]-np.linalg.slogdet(oinv+sinv+ks)[1])
    result+= -.5*(om-sm)@off_sus_inv@(om-sm)+logsumquad(typ,bothk)+np.log(k)-logsumquad(om,offk)-logsumquad(sm,susk)
    return float(np.clip(result, -np.log(np.finfo(float).max), np.log(np.finfo(float).max)))


T=time.time(); rows=[]
for cn,(aa,bb) in COND.items():
 for v in list('aeiou'):
    d=raw[(raw.sesion=='A')&(raw.vocal==v)&raw.hablante.isin(hab)].dropna(subset=aa)
    e=raw[(raw.sesion=='B')&(raw.vocal==v)&raw.hablante.isin(hab)].dropna(subset=bb)
    bg={h:d[d.hablante==h][aa].to_numpy() for h in hab}
    off={h:e[e.hablante==h][bb].to_numpy() for h in hab}
    assert min(map(len,bg.values()))>=5 and min(map(len,off.values()))>=5
    for i in hab:
     for j in hab:
      z=(np.mean([score(bg[i],off[j],bg,i,j,removed_extra=k) for k in hab if k!=i]) if i==j else score(bg[i],off[j],bg,i,j))
      rows.append((cn,v,i,j,z,int(i==j)))
    print('Finished',cn,v,round(time.time()-T,1),'s',flush=True)
res=pd.DataFrame(rows,columns=['condicion','vocal','sus','off','logLR','mismo'])
res.to_csv(BASE/'resultados'/'mvkd_pares_fondo_corregido_python.csv',index=False)
print('output rows',len(res),'duration',time.time()-T)
