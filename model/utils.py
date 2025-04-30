def create_score_normalizer(pos_scores, neg_scores):
    """
    주어진 Positive / Negative 점수 리스트를 기반으로
    정규화 함수 (입력 점수 → [-1, 1]로 변환) 를 반환
    """
    import numpy as np

    # 1. 히스토그램 겹침 기반 기준점 찾기
    hist_range = (min(np.min(pos_scores), np.min(neg_scores)),
                  max(np.max(pos_scores), np.max(neg_scores)))
    bins = 200
    pos_hist, bin_edges = np.histogram(pos_scores, bins=bins, range=hist_range)
    neg_hist, _ = np.histogram(neg_scores, bins=bins, range=hist_range)
    overlap = np.minimum(pos_hist, neg_hist)
    overlap_peak_index = np.argmax(overlap)
    overlap_peak = (bin_edges[overlap_peak_index] + bin_edges[overlap_peak_index + 1]) / 2

    # 2. 전체 범위 기준으로 최대 절댓값 계산
    all_scores = np.concatenate([pos_scores, neg_scores])
    shifted = all_scores - overlap_peak
    max_abs = np.max(np.abs(shifted))

    # 3. 정규화 함수 반환
    def normalize_fn(score):
        return (score - overlap_peak) / max_abs

    return normalize_fn
