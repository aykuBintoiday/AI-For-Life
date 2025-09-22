import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def budget_score(avg_cost, user_budget):
    return max(0.0, 1.0 - abs(avg_cost - user_budget) / (user_budget + 1e-6))

def duration_score(min_days, user_days):
    return 1.0 if min_days <= user_days else max(0.0, 1.0 - (min_days - user_days)/max(1, min_days))

def rank_with_constraints(q_vec, X_sub, avg_costs, min_days, user_budget, user_days, w=(0.6,0.25,0.15)):
    cos = cosine_similarity(q_vec, X_sub).ravel()
    b = np.array([budget_score(c, user_budget) for c in avg_costs])
    d = np.array([duration_score(m, user_days) for m in min_days])
    final = w[0]*cos + w[1]*b + w[2]*d
    order = np.argsort(-final)
    return order, cos[order], b[order], d[order], final[order]
