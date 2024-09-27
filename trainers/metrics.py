from sklearn.metrics import ndcg_score, roc_auc_score
import numpy as np
import torch
import math

def nDCG(labels, scores, k):
    '''
    >>> from sklearn.metrics import ndcg_score
    >>> # we have groud-truth relevance of some answers to a query:
    >>> true_relevance = np.asarray([[10, 0, 0, 1, 5]])
    >>> # we predict some scores (relevance) for the answers
    >>> scores = np.asarray([[.1, .2, .3, 4, 70]])
    >>> ndcg_score(true_relevance, scores)
    0.69...
    >>> scores = np.asarray([[.05, 1.1, 1., .5, .0]])
    >>> ndcg_score(true_relevance, scores)
    0.49...
    >>> # we can set k to truncate the sum; only top k answers contribute.
    >>> ndcg_score(true_relevance, scores, k=4)
    0.35...
    >>> # the normalization takes k into account so a perfect answer
    >>> # would still get 1.0
    >>> ndcg_score(true_relevance, true_relevance, k=4)
    1.0
    '''
    return ndcg_score(y_true=labels, y_score=scores, k=k)


def auc(labels, scores):
    '''
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import roc_auc_score
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    >>> roc_auc_score(y, clf.predict_proba(X)[:, 1])
    0.99...
    >>> roc_auc_score(y, clf.decision_function(X))
    0.99...
    '''
    return roc_auc_score(labels, scores)


def mrr(labels, scores):
    '''
    @ label/ score - numpy
    '''
    mat = np.array([scores, labels]).T
    mat = mat[mat[:, 0].argsort()]
    mat = mat[::-1]
    mrr = 0
    count = 0
    for i, (s, l) in enumerate(mat):
        if l == 1:
            mrr += 1/(i+1)
            count += 1
    return mrr / count

def ils(recommended_news_repr):
    num_user, topk = recommended_news_repr.shape[0], recommended_news_repr.shape[1]
    ils_mat = torch.matmul(recommended_news_repr, recommended_news_repr.permute(0,2,1))
    diag_sum = 0
    for i in range(topk):
        diag_sum += ils_mat[:, i, i].sum()
    ils_score = (ils_mat.sum() - diag_sum) / (topk-1)**2 / num_user
    return ils_score

def ilad(recommended_news_repr):
    topk = recommended_news_repr.shape[1]
    ilad_score = 0
    count = 0
    for i in range(topk):
        for j in range(i+1, topk):
            ilad_score += ((recommended_news_repr[:, i, :] - recommended_news_repr[:, j, :])**2).sum(dim=-1).sqrt().mean()
            count += 1
    return ilad_score / count

def ILAD(vecs):
    num = vecs.shape[0]
    dis = []
    for i in range(num):
        for j in range(i+1, num):
            dis.append(np.sqrt(((vecs[i] - vecs[j])**2).sum()))
    # score = np.dot(vecs, vecs.T)
    # score = (score+1)/2
    # score = score.mean()-1/score.shape[0]
    # score = float(score)
    # score = np.sqrt((1-score**2)+(1-score)**2).mean()
    return sum(dis) / len(dis)