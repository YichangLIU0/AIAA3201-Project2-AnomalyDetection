"""Normal-calibrated reject option. No test labels enter policy construction."""
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import roc_curve
from shared_protocol.metrics import normal_threshold

CONFIG = {
    'version':1, 'threshold':'shared linear95th percentile of20 fixed calibration normals',
    'uncertainty':'absolute raw score distance from the normal-calibrated threshold',
    'main_width':'linear10th percentile of absolute calibration distances from threshold',
    'review_rule':'symmetric closed interval [threshold-width,threshold+width]; reviewed samples are not automatically correct',
    'ablation':'lower-side-only review using the identical width; unchanged detector scores and threshold',
    'curve':'accept highest absolute margins first; accept tied confidence as a group; labels used only to measure errors',
    'fixed_fpr':'best attainable test-ROC FNR at empirical FPR<=.05, no interpolation; evaluation only, never a deployment threshold',
    'risk':'(automatic FP+automatic FN)/number automatically accepted; null at zero acceptance',
    'automatic_fnr':'automatic FN/all test anomalies; REVIEW anomalies separately reported, never counted as TP',
    'random_control':'100 seed-fixed label-blind random acceptance sets of same cardinality, measurement-only comparator',
}


def finite_vector(values, name):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a nonempty finite 1D array')
    return values


@dataclass(frozen=True)
class ReviewPolicy:
    threshold: float
    width: float
    calibration_count: int=20
    width_quantile: float=.10

    @classmethod
    def fit(cls, calibration_scores):
        cal=finite_vector(calibration_scores,'calibration_scores')
        threshold=normal_threshold(cal)
        width=float(np.quantile(np.abs(cal-threshold),.10,method='linear'))
        return cls(threshold,width)

    def decisions(self, scores, mode='symmetric'):
        scores=finite_vector(scores,'test_scores')
        if mode=='none': review=np.zeros(len(scores),dtype=bool)
        elif mode=='symmetric': review=np.abs(scores-self.threshold)<=self.width
        elif mode=='lower_only': review=(scores>=self.threshold-self.width)&(scores<=self.threshold)
        else: raise ValueError('Unknown review mode')
        return np.where(review,'REVIEW',np.where(scores>self.threshold,'NG','OK'))


def counts(scores, labels, policy, accepted):
    scores=finite_vector(scores,'scores'); labels=np.asarray(labels)
    accepted=np.asarray(accepted,dtype=bool)
    if labels.shape!=scores.shape or accepted.shape!=scores.shape or set(np.unique(labels))!={0,1}:
        raise ValueError('Expected aligned binary test labels, scores, and acceptance mask')
    pred=scores>policy.threshold
    fp=int(np.sum(accepted & pred & (labels==0)))
    fn=int(np.sum(accepted & ~pred & (labels==1)))
    tp=int(np.sum(accepted & pred & (labels==1)))
    tn=int(np.sum(accepted & ~pred & (labels==0)))
    review_anomaly=int(np.sum(~accepted & (labels==1)))
    review_normal=int(np.sum(~accepted & (labels==0)))
    n=int(accepted.sum())
    return {'coverage':n/len(scores),'review_rate':1-n/len(scores),
            'risk':(fp+fn)/n if n else None,'n_automatic':n,'n_review':len(scores)-n,
            'automatic_fp':fp,'automatic_fn':fn,'automatic_tp':tp,'automatic_tn':tn,
            'review_anomaly':review_anomaly,'review_normal':review_normal,
            'automatic_fnr':fn/int((labels==1).sum()),'automatic_fpr':fp/int((labels==0).sum()),
            'accepted_anomaly_fnr':fn/(fn+tp) if fn+tp else None,
            'accepted_normal_fpr':fp/(fp+tn) if fp+tn else None,
            'anomaly_accounting_total':fn+tp+review_anomaly}


def risk_coverage(scores, labels, policy):
    scores=finite_vector(scores,'scores'); labels=np.asarray(labels)
    confidence=np.abs(scores-policy.threshold)
    levels=np.unique(confidence)[::-1]
    curve=[{'coverage':0.,'risk':None,'n_automatic':0,'confidence_min':None}]
    for value in levels:
        row=counts(scores,labels,policy,confidence>=value)
        row['confidence_min']=float(value); curve.append(row)
    # Right-step area: empirical error risk at each nonempty accepted fraction.
    aurc=sum((b['coverage']-a['coverage'])*b['risk'] for a,b in zip(curve,curve[1:]))
    return curve,float(aurc)


def evaluate_selective(calibration, scores, labels):
    policy=ReviewPolicy.fit(calibration)
    scores=finite_vector(scores,'scores'); labels=np.asarray(labels)
    policies={}
    for mode in ['none','symmetric','lower_only']:
        decision=policy.decisions(scores,mode)
        policies[mode]=counts(scores,labels,policy,decision!='REVIEW')
    curve,aurc=risk_coverage(scores,labels,policy)
    n=policies['symmetric']['n_automatic']
    generator=np.random.default_rng(1729)
    risks=[]
    for _ in range(100):
        mask=np.zeros(len(scores),bool); mask[generator.permutation(len(scores))[:n]]=True
        risks.append(counts(scores,labels,policy,mask)['risk'])
    random_risk=None if n==0 else {'mean':float(np.mean(risks)),'std':float(np.std(risks)),
                                  'repetitions':100,'acceptance_count':n,'seed':1729}
    fpr,tpr,thresholds=roc_curve(labels,scores,drop_intermediate=False)
    valid=np.flatnonzero(fpr<=.05+1e-12)
    selected=valid[np.argmax(tpr[valid])]
    # Infinite all-OK threshold is a legitimate ROC point but not JSON float.
    threshold=float(thresholds[selected]) if np.isfinite(thresholds[selected]) else None
    return {'policy':asdict(policy),'policies':policies,'risk_coverage':curve,'aurc':aurc,
            'random_review_control':random_risk,
            'fixed_fpr_test_roc':{'requested_fpr_ceiling':.05,'attained_fpr':float(fpr[selected]),
                 'fnr':float(1-tpr[selected]),'threshold_evaluation_only':threshold,
                 'threshold_is_infinite':not bool(np.isfinite(thresholds[selected])),
                 'decision_rule':'score >= ROC threshold; diagnostic only; deploy uses shared score > calibration threshold',
                 'test_labels_used_for_evaluation_only':True}}
