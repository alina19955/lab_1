"""
Лабораторная работа 2, вариант 9: отбор признаков (PCA / SVD / ICA) и кластеризация.

Запуск:  python lab2_variant9.py Lab2_Dataset_PCA.xlsx Lab2_Dataset_Clustering.xlsx
Результаты: figs/f1.png ... f12.png и results.json
"""
import json, os, sys, warnings, random
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.cluster import (KMeans, BisectingKMeans, AgglomerativeClustering, SpectralClustering,
    AffinityPropagation, DBSCAN, HDBSCAN, OPTICS, MeanShift, Birch)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
warnings.filterwarnings("ignore")
SEED=42; V=9; random.seed(SEED); np.random.seed(SEED)
PCA_FILE=sys.argv[1] if len(sys.argv)>1 else "Lab2_Dataset_PCA.xlsx"
CLU_FILE=sys.argv[2] if len(sys.argv)>2 else "Lab2_Dataset_Clustering.xlsx"
os.makedirs("figs",exist_ok=True)
F="figs/"; R={}
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.3})

p=pd.read_excel(PCA_FILE,header=None)
P=p.iloc[2:,(V-1)*6:V*6].dropna(how="all").astype(float); P.columns=list("abcdey"); P["y"]=P["y"].astype(int)
X=P[list("abcde")]; y=P["y"].to_numpy(); R["n_pca"]=len(P); R["classes_pca"]=sorted(set(y.tolist()))
Xs=StandardScaler().fit_transform(X)

# 1.1
C=X.corr(); R["corr"]=C.round(3).to_dict()
fig,ax=plt.subplots(figsize=(5.5,4.6)); im=ax.imshow(C,cmap="coolwarm",vmin=-1,vmax=1); ax.grid(False)
ax.set_xticks(range(5),C.columns); ax.set_yticks(range(5),C.index)
for i in range(5):
    for j in range(5): ax.text(j,i,f"{C.iloc[i,j]:.2f}",ha="center",va="center",fontsize=9)
plt.colorbar(im); ax.set_title("Матрица корреляций (Пирсон)"); fig.savefig(F+"f1.png",dpi=150,bbox_inches="tight"); plt.close()
# 1.2
fig,axs=plt.subplots(5,5,figsize=(9,9)); cols="a b c d e".split(); cm=plt.get_cmap("viridis",4)
for i,ci in enumerate(cols):
    for j,cj in enumerate(cols):
        a=axs[i,j]
        if i==j: a.hist(X[ci],bins=20,color="gray")
        else: a.scatter(X[cj],X[ci],c=y,cmap=cm,s=8)
        if i==4:a.set_xlabel(cj)
        if j==0:a.set_ylabel(ci)
        a.tick_params(labelsize=6)
fig.suptitle("Матрица разброса (цвет = класс y)"); fig.savefig(F+"f2.png",dpi=130,bbox_inches="tight"); plt.close()

def biplot(Z,load,name,title,fn,lab):
    fig,ax=plt.subplots(figsize=(6,5)); s=ax.scatter(Z[:,0],Z[:,1],c=y,cmap=cm,s=14)
    sc=np.abs(Z[:,:2]).max()/np.abs(load[:,:2]).max()*.8
    for k,c in enumerate(cols):
        ax.arrow(0,0,load[k,0]*sc,load[k,1]*sc,color="red",head_width=.08*sc/3,alpha=.8); ax.text(load[k,0]*sc*1.1,load[k,1]*sc*1.1,c,color="red")
    ax.set_xlabel(lab+"1"); ax.set_ylabel(lab+"2"); ax.set_title(title); plt.colorbar(s,label="класс y",ticks=range(4)); fig.savefig(F+fn,dpi=150,bbox_inches="tight"); plt.close()

def sig_plot(vals,names,title,fn,ylabel,cum=True):
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(names,vals,color="steelblue")
    if cum: ax2=ax.twinx(); ax2.plot(names,np.cumsum(vals),"o-",color="darkorange"); ax2.set_ylim(0,1.05); ax2.set_ylabel("кумулятивная доля"); ax2.grid(False)
    ax.set_ylabel(ylabel); ax.set_title(title); fig.savefig(F+fn,dpi=150,bbox_inches="tight"); plt.close()

# 1.3.1 PCA
pca=PCA(n_components=5).fit(Xs); ev=pca.explained_variance_ratio_; Zp=pca.transform(Xs)
R["pca"]=ev.tolist(); R["pca_load"]=pca.components_.T.round(3).tolist()
sig_plot(ev,[f"PC{i+1}" for i in range(5)],"Значимость компонент PCA","f3.png","доля дисперсии")
biplot(Zp,pca.components_.T,"PCA","PCA Biplot (PC1 vs PC2)","f4.png","PC")
# SVD
svd=TruncatedSVD(n_components=5,algorithm="randomized",n_iter=20,random_state=SEED).fit(Xs); es=svd.explained_variance_ratio_; Zs=svd.transform(Xs)
R["svd"]=es.tolist()
sig_plot(es,[f"SVD{i+1}" for i in range(5)],"Значимость компонент SVD","f5.png","доля дисперсии")
biplot(Zs,svd.components_.T,"SVD","SVD Biplot (SVD1 vs SVD2)","f6.png","SVD")
# ICA
ica=FastICA(n_components=4,whiten="unit-variance",random_state=SEED,max_iter=2000).fit(Xs); Zi=ica.transform(Xs)
k=np.abs(kurtosis(Zi,axis=0)); kn=k/k.sum(); R["ica_kurt_raw"]=k.tolist(); R["ica"]=kn.tolist()
sig_plot(kn,[f"IC{i+1}" for i in range(4)],"Значимость компонент ICA (по |куртозису|)","f7.png","|куртозис| (норм.)",cum=False)
biplot(Zi,ica.mixing_,"ICA","ICA Biplot (IC1 vs IC2)","f8.png","IC")
# 1.4 classification
def cls_var(v): return ([i+1 for i,x in enumerate(v) if x>=.10],[i+1 for i,x in enumerate(v) if .01<=x<.10],[i+1 for i,x in enumerate(v) if x<.01])
def cls_ica(v): return ([i+1 for i,x in enumerate(v) if x>=.20],[i+1 for i,x in enumerate(v) if .10<=x<.20],[i+1 for i,x in enumerate(v) if x<.10])
R["cls"]={"PCA":cls_var(ev),"SVD":cls_var(es),"ICA":cls_ica(kn)}

# ---------- Section 2
c=pd.read_excel(CLU_FILE,header=None)
D=c.iloc[2:,(V-1)*3:V*3].dropna(how="all").astype(float); D.columns=["a","b","y"]; yt=D["y"].astype(int).to_numpy()
Xc=StandardScaler().fit_transform(D[["a","b"]]); R["n_clu"]=len(D); R["k"]=int(len(set(yt)))
K=R["k"]
def metrics(lab):
    m=lab!=-1; ncl=len(set(lab[m].tolist())); noise=int((~m).sum())
    ari=adjusted_rand_score(yt,lab)
    sil=silhouette_score(Xc[m],lab[m]) if ncl>=2 and m.sum()>ncl else float("nan")
    return ncl,noise,ari,sil
# density tuning
def best(make,grid):
    res=[]
    for g in grid:
        try:
            lab=make(**g).fit_predict(Xc)
            if (lab==-1).mean()>0.10: continue   # отбрасываем вырожденные варианты (шум > 10% точек)
            res.append((adjusted_rand_score(yt,lab),g,lab))
        except Exception: pass
    return max(res,key=lambda t:t[0])
db_grid=[dict(eps=e,min_samples=m) for e in np.round(np.arange(.1,1.01,.05),2) for m in (3,5,8,10)]
hd_grid=[dict(min_cluster_size=m,min_samples=s) for m in (5,10,15,20,30) for s in (None,3,5,10)]
op_grid=[dict(min_samples=m,xi=x,min_cluster_size=mc) for m in (5,10,15) for x in (.02,.05,.1,.2) for mc in (.05,.1)]
db_b=best(DBSCAN,db_grid); hd_b=best(HDBSCAN,hd_grid); op_b=best(OPTICS,op_grid)
R["tuned"]={"DBSCAN":{k:float(v) for k,v in db_b[1].items()},"HDBSCAN":{k:(None if v is None else int(v)) for k,v in hd_b[1].items()},"OPTICS":{k:float(v) for k,v in op_b[1].items()}}
algos={
 "K-Means":KMeans(K,n_init=10,random_state=SEED),
 "BisectingKMeans":BisectingKMeans(K,random_state=SEED),
 "GaussianMixture":GaussianMixture(K,random_state=SEED),
 "Ward":AgglomerativeClustering(K,linkage="ward"),
 "Agglomerative (average)":AgglomerativeClustering(K,linkage="average"),
 "SpectralClustering (n_neighbors=20)":SpectralClustering(K,affinity="nearest_neighbors",n_neighbors=20,random_state=SEED),
 "AffinityPropagation":AffinityPropagation(random_state=SEED),
 "MeanShift":MeanShift(),
 "BIRCH":Birch(n_clusters=K),
 f"DBSCAN (eps={db_b[1]['eps']}, ms={db_b[1]['min_samples']})":DBSCAN(**db_b[1]),
 "HDBSCAN":HDBSCAN(**hd_b[1]),
 "OPTICS":OPTICS(**op_b[1]),
}
labels={}; rows=[]
for n,a in algos.items():
    lab=a.fit_predict(Xc) if hasattr(a,"fit_predict") else a.fit(Xc).predict(Xc); labels[n]=np.asarray(lab)
    ncl,noise,ari,sil=metrics(labels[n]); rows.append(dict(name=n,k=ncl,noise=noise,ari=ari,sil=sil))
# default DBSCAN for reference
dflt=DBSCAN(eps=.5,min_samples=5).fit_predict(Xc); R["dbscan_default"]=dict(zip(["k","noise","ari","sil"],metrics(dflt)))
rows.sort(key=lambda r:(-round(r["ari"],9),-(r["sil"] if r["sil"]==r["sil"] else -1)))
R["table"]=rows
mx=max(r["ari"] for r in rows); grp=[r["name"] for r in rows if abs(r["ari"]-mx)<1e-9 and r["noise"]==0]
R["tie_group"]=grp; R["best"]="K-Means" if "K-Means" in grp else grp[0]
# проверка: какая точка помечена шумом в DBSCAN по умолчанию и совпадает ли она с ошибкой эталона
km=labels["K-Means"]
cts=pd.crosstab(yt,km); mapping={c:cts[c].idxmax() for c in cts.columns}
mism=[i for i in range(len(yt)) if mapping[km[i]]!=yt[i]]
R["mismatch_idx"]=mism; R["dbscan_default_noise_idx"]=np.where(dflt==-1)[0].tolist()
sw={}
for nn in (5,10,15,20,30,50):
    l=SpectralClustering(K,affinity="nearest_neighbors",n_neighbors=nn,random_state=SEED).fit_predict(Xc); sw[nn]=float(adjusted_rand_score(yt,l))
R["spectral_sweep"]=sw
# fig9
fig,axs=plt.subplots(3,4,figsize=(14,9.5))
for a,(n,lab) in zip(axs.ravel(),labels.items()):
    a.scatter(Xc[:,0],Xc[:,1],c=lab,cmap="tab10",s=10); m=lab==-1
    if m.any(): a.scatter(Xc[m,0],Xc[m,1],c="k",marker="x",s=18)
    r=[t for t in rows if t["name"]==n][0]; a.set_title(f"{n}\nARI={r['ari']:.3f}",fontsize=9)
fig.suptitle("Результаты 12 алгоритмов кластеризации (× — шум)"); fig.tight_layout(); fig.savefig(F+"f9.png",dpi=120); plt.close()
# fig10 eps scan
es_=np.round(np.arange(.1,1.51,.05),2); sc=[]
for e in es_:
    l=DBSCAN(eps=e,min_samples=5).fit_predict(Xc); sc.append((len(set(l[l!=-1].tolist())),int((l==-1).sum()),adjusted_rand_score(yt,l)))
sc=np.array(sc); R["eps_scan"]=dict(eps=es_.tolist(),k=sc[:,0].tolist(),noise=sc[:,1].tolist(),ari=sc[:,2].tolist())
fig,ax=plt.subplots(1,2,figsize=(11,4)); ax[0].plot(es_,sc[:,1],"o-",color="crimson"); ax[0].set_xlabel("eps"); ax[0].set_ylabel("число шумовых точек"); ax[0].set_title("Шум vs eps (min_samples=5)")
ax[1].plot(es_,sc[:,2],"o-"); ax[1].set_xlabel("eps"); ax[1].set_ylabel("ARI"); ax[1].set_title("ARI vs eps"); fig.savefig(F+"f10.png",dpi=140,bbox_inches="tight"); plt.close()
# fig11/12
bl=labels[R["best"]]
fig,ax=plt.subplots(figsize=(6,4.8)); ax.scatter(Xc[:,0],Xc[:,1],c=bl,cmap="tab10",s=16); ax.set_title(f"Лучший алгоритм: {R['best']}"); ax.set_xlabel("a (станд.)"); ax.set_ylabel("b (станд.)"); fig.savefig(F+"f11.png",dpi=150,bbox_inches="tight"); plt.close()
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
ax[0].scatter(Xc[:,0],Xc[:,1],c=yt,cmap="tab10",s=16); ax[0].set_title("Эталонная разметка y")
ax[1].scatter(Xc[:,0],Xc[:,1],c=bl,cmap="tab10",s=16); ax[1].set_title(f"{R['best']} (ARI={rows[0]['ari']:.3f})")
fig.savefig(F+"f12.png",dpi=150,bbox_inches="tight"); plt.close()
# raw data picture for context
json.dump(R,open("results.json","w"),ensure_ascii=False,indent=1,default=float)
print(json.dumps({k:R[k] for k in ["cls","tuned","dbscan_default","best","tie_group","mismatch_idx","dbscan_default_noise_idx","spectral_sweep"]},ensure_ascii=False))
for r in rows: print(r)
