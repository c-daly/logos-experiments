# Clustering parameter-sweep report

- source: live-graph-capture
- entities: 183  edges: 107
- n_domains (node ground truth): 6
- deps: sklearn=True hdbscan=True umap=True

## Node configs (ranked by ARI vs domain)

| scheme | algorithm | preproc | min | k_mode | n_cl | cover | ARI | purity | combined |
|---|---|---|---|---|---|---|---|---|---|
| name+ctx:concat:a0.5 | agglomerative_avg | pca50 | 2 | n_domains | 6 | 1.000 | 0.956 | 0.981 | 1.202 |
| name+ctx:concat:a0.5 | agglomerative_avg | pca50 | 3 | n_domains | 6 | 1.000 | 0.956 | 0.981 | 1.202 |
| name+ctx:concat:a0.5 | agglomerative_avg | pca50 | 5 | n_domains | 6 | 1.000 | 0.956 | 0.981 | 1.202 |
| name+ctx:concat:a0.3 | agglomerative_avg | raw | 2 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.3 | agglomerative_avg | raw | 3 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.3 | agglomerative_avg | raw | 5 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.3 | agglomerative_avg | l2norm | 2 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.3 | agglomerative_avg | l2norm | 3 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.3 | agglomerative_avg | l2norm | 5 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.3 | agglomerative_avg | pca50 | 2 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.3 | agglomerative_avg | pca50 | 3 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.3 | agglomerative_avg | pca50 | 5 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_avg | raw | 2 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_avg | raw | 3 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_avg | raw | 5 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_complete | raw | 2 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_complete | raw | 3 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_complete | raw | 5 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_avg | l2norm | 2 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_avg | l2norm | 3 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_avg | l2norm | 5 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_complete | l2norm | 2 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_complete | l2norm | 3 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| name+ctx:concat:a0.5 | agglomerative_complete | l2norm | 5 | n_domains | 6 | 1.000 | 0.914 | 0.963 | 1.154 |
| context | agglomerative_avg | raw | 2 | n_domains | 6 | 1.000 | 0.876 | 0.944 | 1.112 |

## Edge configs (ranked by relation-label purity)

| scheme | algorithm | preproc | n_cl | rel_purity | merge_ratio | endpoint_homog |
|---|---|---|---|---|---|---|
| relationship_label | agglomerative_avg | raw | 80 | 1.000 | 1.000 | 1.000 |
| relationship_label | kmeans | raw | 80 | 1.000 | 1.000 | 1.000 |
| relationship_label | agglomerative_avg | l2norm | 80 | 1.000 | 1.000 | 1.000 |
| relationship_label | kmeans | l2norm | 80 | 1.000 | 1.000 | 1.000 |
| label+ctx:concat:a0.7 | agglomerative_avg | raw | 73 | 0.990 | 1.096 | 1.000 |
| label+ctx:concat:a0.7 | kmeans | raw | 73 | 0.990 | 1.096 | 1.000 |
| label+ctx:concat:a0.7 | agglomerative_avg | l2norm | 73 | 0.990 | 1.096 | 1.000 |
| label+ctx:concat:a0.7 | kmeans | l2norm | 73 | 0.990 | 1.096 | 1.000 |
| label+ctx:weighted:a0.7 | agglomerative_avg | raw | 73 | 0.990 | 1.096 | 1.000 |
| label+ctx:weighted:a0.7 | kmeans | raw | 73 | 0.990 | 1.096 | 1.000 |
| label+ctx:weighted:a0.7 | agglomerative_avg | l2norm | 73 | 0.990 | 1.096 | 1.000 |
| label+ctx:weighted:a0.7 | kmeans | l2norm | 73 | 0.990 | 1.096 | 1.000 |
| label+ctx:concat:a0.3 | agglomerative_avg | raw | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:concat:a0.3 | kmeans | raw | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:concat:a0.3 | agglomerative_avg | l2norm | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:concat:a0.3 | kmeans | l2norm | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:concat:a0.5 | agglomerative_avg | raw | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:concat:a0.5 | kmeans | raw | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:concat:a0.5 | agglomerative_avg | l2norm | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:concat:a0.5 | kmeans | l2norm | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:weighted:a0.3 | agglomerative_avg | raw | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:weighted:a0.3 | kmeans | raw | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:weighted:a0.3 | agglomerative_avg | l2norm | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:weighted:a0.3 | kmeans | l2norm | 73 | 0.835 | 1.096 | 1.000 |
| label+ctx:weighted:a0.5 | agglomerative_avg | raw | 73 | 0.835 | 1.096 | 1.000 |
