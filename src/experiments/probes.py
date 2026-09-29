"""Matched regularized multinomial linear decoding across layers."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fit_probe_curves(X,labels,splits,seed=0,C_values=(.01,.1,1,10)):
    """X [N,L,H], labels [N] candidate ids, splits [N] train/validation/test."""
    train=splits=="train"; val=splits=="validation"; test=splits=="test"; out=[]
    for layer in range(X.shape[1]):
        scaler=StandardScaler().fit(X[train,layer]); xtrain=scaler.transform(X[train,layer]); xv=scaler.transform(X[val,layer]); xt=scaler.transform(X[test,layer]); best=None
        for c in C_values:
            model=LogisticRegression(C=c,max_iter=1000,random_state=seed).fit(xtrain,labels[train]); acc=(model.predict(xv)==labels[val]).mean()
            if best is None or acc>best[0]: best=(acc,c,model)
        model=LogisticRegression(C=best[1],max_iter=1000,random_state=seed).fit(xtrain,labels[train]); rng=np.random.default_rng(seed+layer); shuffled=labels[train].copy(); rng.shuffle(shuffled); null=LogisticRegression(C=best[1],max_iter=1000,random_state=seed).fit(xtrain,shuffled)
        out.append({"layer":layer,"seed":seed,"C":best[1],"train_accuracy":float((model.predict(xtrain)==labels[train]).mean()),"validation_accuracy":float(best[0]),"test_accuracy":float((model.predict(xt)==labels[test]).mean()),"shuffled_label_test_accuracy":float((null.predict(xt)==labels[test]).mean()),"chance":float(1/len(np.unique(labels[train])))})
    return out
