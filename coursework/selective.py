"""Selective review keeps the original NG threshold; REVIEW is not a correct label."""
import numpy as np

def decisions(scores,review_threshold,ng_threshold):
 if review_threshold>ng_threshold:raise ValueError('Review threshold must not exceed NG threshold')
 x=np.asarray(scores);result=np.full(x.shape,'OK',dtype='<U6')
 result[x>review_threshold]='REVIEW';result[x>ng_threshold]='NG'
 return result
