import streamlit as st, pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
st.title("Customer Segmentation Demo")
df=pd.DataFrame({"AnnualSpend":[120,140,160,650,700,720,300,320,340],"Visits":[2,3,4,10,11,12,6,7,8]})
k=st.slider("Clusters",2,5,3); X=StandardScaler().fit_transform(df)
df["Cluster"]=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X).astype(str)
st.dataframe(df); fig,ax=plt.subplots()
for name,g in df.groupby("Cluster"): ax.scatter(g.AnnualSpend,g.Visits,label=name)
ax.set_xlabel("Annual spend"); ax.set_ylabel("Visits"); ax.legend(); st.pyplot(fig)
st.caption("Synthetic example data only.")
