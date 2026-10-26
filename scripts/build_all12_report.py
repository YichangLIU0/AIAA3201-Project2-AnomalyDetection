"""Create the all-twelve report from complete, checked experiment summaries."""
from pathlib import Path
import json,re
import numpy as np
r=Path(__file__).resolve().parents[1];out=r/'outputs/coursework';report=r/'reports'
summary=json.loads((out/'all12_summary.json').read_text())
def row(method,scope='all12',shots=8):return next(x for x in summary if x['method']==method and x['scope']==scope and x['shots']==shots)
def val(method,key,scope='all12',shots=8):return f'{row(method,scope,shots)[key]:.3f}'
def percent(method,key,scope='all12'):return f'{100*row(method,scope)[key]:.2f}'
methods=['coreset25','hi448','multiscale','core224_peak','hi448_peak']
names={'full':'DINO full','random25':'Random25','pooled25':'Pooled25','coreset25':'Coreset224','hi448':'Coreset448','multiscale':'Two-scale','core224_peak':'224 peak','hi448_peak':'448 peak','template_ecc':'Template ECC','autoencoder':'Autoencoder','anomalyclip':'AnomalyCLIP'}
keys=['image_auroc','image_ap','pixel_auroc','pixel_ap','aupro_03','image_fpr','image_fnr']
lines=[r'\begin{center}\scriptsize',r'\tablecaption{All twelve categories: category-macro mean $\pm$ sample standard deviation across three reference draws. AnomalyCLIP uses one fixed checkpoint. All operating thresholds use 40 additional normal images per category. Lower FPR/FNR is better.}',r'\resizebox{\textwidth}{!}{\begin{tabular}{lrccccccc}\toprule',r'Method & Shots & I-AUROC & I-AP & P-AUROC & P-AP & AUPRO$_{.3}$ & FPR & FNR \\\midrule']
for x in [a for a in summary if a['scope']=='all12']:
 cells=[names[x['method']],str(x['shots'])]
 cells += [f"{x[k]:.3f}" if x['method']=='anomalyclip' else f"${x[k]:.3f}\\pm{x[k+'_std']:.3f}$" for k in keys]
 lines.append(' & '.join(cells)+r' \\')
lines += [r'\bottomrule\end{tabular}}\label{tab:main}\end{center}']
(report/'results_tables.tex').write_text('\n'.join(lines))
manifest=[json.loads(p.read_text()) for p in (out/'manifests').glob('*.json')];assert len(manifest)==12
ntest=sum(len(m['test']) for m in manifest)
paired=json.loads((out/'lowmiss_paired_analysis.json').read_text())
size=json.loads((out/'lowmiss_size_results.json').read_text())
classlines=[]
for category in dict.fromkeys(x['category'] for x in paired['per_run']):
 values=[]
 for method,key in [('coreset25','image_fnr'),('hi448','image_fnr'),('hi448_peak','image_fnr'),('hi448_peak','image_fpr')]:
  rr=[x[key] for x in paired['per_run'] if x['category']==category and x['method']==method];values.append(f'{np.mean(rr):.3f}')
 classlines.append(category.replace('_',r'\_')+' & '+' & '.join(values)+r' \\')
classtable='\n'.join(classlines)
review=json.loads((out/'selective_review_summary.json').read_text())
benchmark=json.loads((out/'online_benchmark.json').read_text())
costtable='\n'.join(f"{names[x['method']]} & {1000*x['mean_seconds']:.1f} & {1000*x['p95_seconds']:.1f} & {x['max_gpu_allocated_mb']:.1f}"+r' \\' for x in benchmark['summary'])
scopes=[]
for scope,label in [('original6','Original six'),('additional6','Additional six'),('all12','All twelve')]:
 for method in ['coreset25','hi448','multiscale']:
  scopes.append(f"{label} & {names[method]} & {val(method,'image_auroc',scope)} & {val(method,'image_fpr',scope)} & {val(method,'image_fnr',scope)}"+r' \\')
scopetable='\n'.join(scopes)
counts={k:sum(x[k] for x in paired['paired_against_core224'] if x['method']=='hi448') for k in ['fn_fixed','fn_introduced','fp_fixed','fp_introduced']}
sizevalues={}
for method in ['coreset25','hi448']:
 rr=[x for x in size if x['method']==method and x['size']=='tiny'];sizevalues[method]=100*sum(x['missed'] for x in rr)/sum(x['count'] for x in rr)
rv=next(x for x in review if x['method']=='hi448' and x['normal_quantile']==.9)
old=(report/'main.tex').read_text(encoding='utf-8')
backup=r/'outputs/report_before_all12.tex'
if not backup.exists():backup.write_text(old,encoding='utf-8')
preamble=old[:old.index('\\title{')]
if r'\newcommand{\tablecaption}' not in preamble:preamble+=r'\makeatletter\newcommand{\tablecaption}[1]{\def\@captype{table}\caption{#1}}\makeatother'+'\n'
text=r'''\title{Reducing Missed Small Defects in Few-Shot Industrial Inspection:\\A Controlled Study across Twelve VisA Categories}
@AUTHOR@
\begin{document}
\maketitle
@FOOTNOTE@
\begin{abstract}
We study few-shot industrial anomaly detection on all twelve VisA categories, using three reference draws and 1, 2, 4, and 8 normal reference images. Forty additional disjoint normal images per category calibrate decisions without test-label threshold tuning. We compare aligned templates, a compact autoencoder, a released AnomalyCLIP checkpoint, and frozen DINOv2 patch matching. Our concrete failure diagnosis is that small defects are frequently missed despite usable ranking metrics. We test a 448-pixel input, fixed-budget two-scale fusion, and an exploratory image-score aggregation ablation. At the same 6 MiB stored-bank budget, changing from 224 to 448 gives macro FNR @CORE_FNR@\% versus @HI_FNR@\%, with FPR @CORE_FPR@\% versus @HI_FPR@\%. We report both repaired and newly introduced errors, localization metrics, reference variation, resource costs, and selective review without counting deferred decisions as detections. These are controlled coursework adaptations of existing methods, not a new anomaly-detection algorithm or a deployment guarantee. Code: \url{https://github.com/YichangLIU0/AIAA3201-Project2-AnomalyDetection}.
\end{abstract}
\section{Introduction}
A useful inspection system must avoid passing defective products while controlling false alarms on normal appearance variation. A high ranking score does not establish a low missed-defect rate at a deployed threshold. In few-shot inspection, a small reference set may fail to represent normal geometry, orientation and texture; small defects can also be weakly represented by a coarse patch grid.

Our initial six-category study exposed this distinction. Coverage-preserving memory selection improved ranking over random compression but increased the calibrated miss rate. We therefore investigate a specific question: can finer input resolution reduce missed small defects without enlarging the stored memory bank? A two-scale control tests whether combining resolutions helps at the same total budget, and a separate scoring ablation tests the choice of image-level aggregation.

We extend the evaluation to the six other VisA categories to include materially different appearances, including printed circuit boards. The extension is a generalization check, not a blinded holdout: development decisions were informed by earlier results. We contribute a reproducible protocol, controlled comparisons and inspectable error evidence. All experimental numbers are measured locally rather than copied from papers.
\section{Related Work}
MVTec AD established an industrial anomaly benchmark with pixel annotations~\cite{mvtec}; VisA supplies this project's data~\cite{visa}. PaDiM models pretrained patch distributions~\cite{padim}, while PatchCore uses nominal patch memory and coverage-based selection~\cite{patchcore}. Our greedy selection adapts an existing principle rather than proposing a novel coreset algorithm.

DRAEM combines reconstruction and discriminative learning~\cite{draem}; CutPaste uses synthetic transformations for representation learning~\cite{cutpaste}. Our plain autoencoder implements neither method. DINOv2 provides pretrained visual features~\cite{dinov2}, and AnomalyDINO applies them to few-shot patch matching~\cite{anomalydino}. AnomalyCLIP learns object-agnostic prompts for anomaly detection~\cite{anomalyclip}; its released checkpoint is an additional comparison with different pretraining resources. EfficientAD motivates measuring computation~\cite{efficientad}, but published latency is not transferable to this laptop. MVTec AD 2 highlights challenging acquisition conditions~\cite{mvtec2}; it is not evaluated here.
\clearpage
\section{Data and Evaluation Protocol}
\subsection{All categories and fixed reference draws}
We use the official VisA one-class split, evaluating all @NTEST@ test images across candle, capsules, cashew, chewinggum, fryum, pipe\_fryum, macaroni1, macaroni2, and pcb1--pcb4. The first six are the required course subset; the remaining six extend coverage. No test image is omitted for speed or appearance. Separate results for the two groups prevent a change in category composition from masquerading as an improvement.

For each category, seed 20261024 reserves 40 normal training images for calibration. They are excluded from the reference pool and autoencoder training. Seeds 0, 1 and 2 draw eight distinct references from the remaining normals. The 1-, 2- and 4-shot sets are nested prefixes of the same eight-reference draw. Draws may overlap; they are not independent datasets. Manifests record all paths and the audit checks reference/calibration separation.

This is a few-reference protocol with additional normal calibration data, not a strict $K$-total-image protocol. Test labels are used for evaluation and retrospective diagnosis only. We do not optimize a threshold using test F1, learn fusion weights from test labels, or select a training epoch on test metrics.
\subsection{Resolution and labels}
The original pipeline uses square bicubic resizing to $224\times224$. The higher-resolution variant reads the original image and resizes directly to $448\times448$; it does not enlarge a cached 224 image. All pixel metrics use the same $224\times224$ ground-truth masks resized by nearest neighbor. A 448 map is interpolated back to this common evaluation grid. Thus the comparison tests feature resolution while keeping metric resolution fixed; it does not measure native-resolution segmentation quality.

An image is anomalous if its resized mask has any positive pixel. A separate audit checks agreement with the official good/bad folders. Defect-size analysis uses the fraction of positive pixels in this common mask. Its bins, at most 0.5\%, 0.5--2\%, and above 2\%, are descriptive and do not establish a causal role for defect size independent of category or defect type.
\subsection{Metrics and averaging}
Image AUROC and average precision measure ranking. Pixel AUROC, pixel AP, and AUPRO measure localization. AUPRO gives equal weight to four-connected ground-truth regions, plots mean region overlap against background-pixel FPR, integrates to FPR 0.3 with boundary interpolation, and divides by 0.3. Background pixels include normal images. Pixel AP is reported because background dominance can make pixel AUROC look reassuring even when localization is weak.

Every category has equal weight. For each reference seed, we average category metrics, then report the mean and sample standard deviation of the three seed means. This describes reference-draw sensitivity on a fixed benchmark, not population uncertainty or a statistical significance test. The fixed AnomalyCLIP checkpoint has one evaluation, not artificial variation from repeated deterministic runs.
\subsection{Normal-only operating thresholds}
For sorted calibration image scores $c_{(1)}\leq\cdots\leq c_{(40)}$, the nominal 5\% false-positive target uses
\begin{equation}
\tau=c_{(\lceil(40+1)(1-0.05)\rceil)}=c_{(39)}.
\end{equation}
A score strictly above $\tau$ is NG; equality is OK. Each method uses its own normal scores under the same rule. The actual held-out FPR need not be 5\%: finite calibration data and score-distribution shifts matter. We report FPR and FNR together rather than claiming equal empirical FPR across methods.

The pixel display threshold is the 99.5th percentile of normal calibration-map pixels. It is separate from image decisions and does not control image-level false alarms. All ranking metrics are computed from continuous scores.
\subsection{Development scope}
The resolution, fixed memory budgets, reference-only fusion scales, and review quantiles were documented before the main resolution comparison. Peak-of-smoothed-map aggregation was added during development on the original categories and is explicitly exploratory. Neither the additional classes nor the whole study should be described as a preregistered independent confirmation. A separate brightness supplement uses 54 settings on the first ten good and ten bad images per original category, RGB factors 0.8/1.0/1.2, and seed-0 full/random/pooled banks with unchanged clean thresholds. It does not test the new methods or establish real acquisition robustness.
\clearpage
\section{Methods and Controlled Comparisons}
\subsection{Appearance and reconstruction baselines}
The template baseline generates eight rotations per reference at $45^\circ$ increments. It selects a template using RGB squared difference at $28\times28$, then estimates Euclidean alignment with ECC, at most 40 iterations and tolerance $10^{-4}$. Failed alignment falls back to the unaligned template. The map is mean absolute RGB residual divided by 255, smoothed with $\sigma=2$; the image score averages its highest one percent of pixels.

The autoencoder has three stride-two convolutions with 16, 32 and 64 channels, followed by three transposed convolutions and a sigmoid. It uses the same 64 augmented eight-shot reference views, MSE, Adam at 0.001, batch size eight, and 15 fixed epochs. No anomaly labels or test-based stopping are used. Its smoothed residual map and top-one-percent score parallel the template baseline. It is deliberately small; poor results do not establish that reconstruction methods generally fail.
\subsection{Frozen patch matching}
We use official DINOv2 ViT-S/14 weights with the released AnomalyDINO feature and PCA query-mask routines. The model is frozen, inference is FP32, and 224/448 inputs produce $16\times16$/$32\times32$ grids of 384-dimensional vectors. An empty PCA foreground mask falls back to all patches. Both references and query vectors are normalized. Exact CPU FAISS nearest-neighbor search gives half the squared distance, equivalent to cosine distance for normalized vectors.

Masked-out query patches have zero score. The standard image score averages the largest one percent of patch distances, rounded down with a minimum of one. Patch maps are interpolated to 224 and Gaussian-smoothed with $\sigma=4$. Our implementation adapts sampling, loading, calibration and evaluation; it is not an exact reproduction of the published paper's full protocol.
\subsection{Memory selection and fair budgets}
Eight references with eight rotations contain 16,384 vectors at 224, occupying 24 MiB in FP32. Random25 retains 4096 original vectors; Pooled25 averages each $2\times2$ group before normalization. Coreset224 also retains 4096 original vectors, selected by greedy coverage in a fixed Gaussian projection from 384 to 128 dimensions, seeded by reference seed plus 2000. The first center is furthest from the projected mean; subsequent centers maximize distance to the nearest selected center. Selection sees references only.

At 448, the pool contains 65,536 vectors. Coreset448 selects 4096 by the same rule. Both single-scale compressed banks occupy 6 MiB. Two-scale fusion stores the first 2048 vectors of each scale's greedy order, again 4096 vectors and 6 MiB in total. Equal stored size does not imply equal feature-extraction or bank-building cost.
\subsection{Reference-only two-scale fusion}
Distances at different scales need not have comparable magnitudes. For each scale, we hold out one reference image at a time, remove all selected vectors originating from any rotation of that image, and score its unrotated view. The scale factor is the median of these eight leave-one-image-out scores, floored at $10^{-6}$. Each half-bank map and score is divided by its scale factor, then combined by elementwise maximum. The final threshold is independently fitted on the same 40 calibration normals. Low-only and high-only half-bank score controls are saved to separate fusion effects from reduced per-scale capacity.
\subsection{Exploratory score aggregation}
For both Coreset224 and Coreset448, a peak variant uses the maximum of the final smoothed map as its image score. Its normal calibration scores are recomputed the same way. Maps, banks and all localization metrics remain identical to the parent. This comparison tests aggregation without training another model; it also changes the score's sensitivity to smoothing. We do not interpret it as a pure causal test of averaging alone.
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/lowmiss_pipeline.pdf}
\caption{Single-scale path. Fusion runs both resolutions with half-size banks; peak aggregation changes only the image score. Calibration normals are excluded from reference construction.}
\end{figure}
\subsection{AnomalyCLIP comparison}
The released checkpoint intended for VisA evaluation and ViT-L/14 backbone are evaluated with 224 inputs, the released selected feature layer, and common pixel metrics. There is no target-category gradient update, but normal calibration uses target data. Different backbone capacity and auxiliary learning prevent interpreting this as an equal-pretraining-resource comparison.
\clearpage
\twocolumn[{
\section{Quantitative Results}
\input{results_tables.tex}
}]
Table~\ref{tab:main} reports the complete twelve-category comparison. At eight shots, full-memory DINO gives image AUROC @FULL_AUC@, compared with @ECC_AUC@ for aligned templates, @AE_AUC@ for the compact autoencoder, and @CLIP_AUC@ for the released AnomalyCLIP checkpoint. These are local results under the stated adaptations, not claims about the methods' published performance.
\subsection{Reference budget and category coverage}
Full-memory image AUROC is @ONE_AUC@ at one reference and @FULL_AUC@ at eight. Figure~\ref{fig:shots} shows all budgets and separates the required and additional groups. A mean trend does not imply monotonic improvement for every category and seed. Complete per-run records allow this distinction to be checked.
\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{figures/all12_shot_curves.pdf}
\caption{Full-memory DINO across nested reference budgets. Original six, additional six, and all twelve are shown separately. Error bars are sample standard deviations over three reference draws.}\label{fig:shots}
\end{figure}
\subsection{Memory reduction}
At eight shots, Random25 and Pooled25 have image AUROC @RANDOM_AUC@ and @POOL_AUC@, respectively, versus @CORE_AUC@ for Coreset224. All three compressed banks store 6 MiB. Random selection and greedy selection retain original vectors; pooling changes their representations as well as count. Consequently this comparison tests three practical compression choices, not feature-count reduction in isolation.

The original-six coverage diagnosis remains instructive: the mean worst reference cosine distance was 0.288 for random sampling and 0.111 for coreset selection, yet FNR increased from 28.78\% to 30.33\%. Better reference coverage is therefore not sufficient evidence of fewer missed anomalies after recalibration. Coverage concerns the reference pool, not a bound on unseen anomaly separation.
\subsection{Operating points matter}
Coreset224's all-twelve FNR is @CORE_FNR@\% at FPR @CORE_FPR@\%. Coreset448 changes these to @HI_FNR@\% and @HI_FPR@\%. These jointly reported outcomes address the project goal more directly than ranking alone. They are not a matched empirical-FPR experiment: the normal quantile rule is matched, while realized FPR can differ.

Two-scale fusion gives FNR @FUSE_FNR@\% and FPR @FUSE_FPR@\%. A more complex combination need not improve on the higher-resolution component. Its 2048-vector component banks are smaller than either single-scale bank; saved half-bank controls help interpret this tradeoff. The 224-only and 448-only half banks have FNR 44.94\% and 32.69\%, at FPR 5.17\% and 5.28\%. Fusion is therefore worse than the high-only half-bank control as well as the full high-resolution bank; reduced component capacity alone does not explain its deficit. We retain negative results rather than changing fusion weights after inspecting test labels.
\clearpage
\section{Diagnosing and Reducing Missed Defects}
\subsection{Concrete failure and intervention}
In the original-six Coreset224 diagnosis, 512 of 1317 anomalous image--seed events with mask area at most 0.5\% were missed (38.9\%), versus one of 141 above 2\% (0.7\%). A coarse grid is a plausible contributor, but category composition and defect appearance confound that association. The higher-resolution intervention tests whether a practical change helps under matched references, bank bytes, masking, calibration and evaluation masks.
\begin{table}[h]\centering\scriptsize
\caption{Scope-separated eight-shot results. Each entry averages three seeds. Both FPR and FNR are shown.}
\begin{tabular}{llrrr}\toprule
Scope & Method & AUROC & FPR & FNR\\\midrule
@SCOPE_TABLE@
\bottomrule\end{tabular}
\end{table}
\subsection{Paired error accounting}
Across all twelve categories and three draws, Coreset448 repairs @FN_FIXED@ Coreset224 false-negative events and introduces @FN_NEW@ new ones. It repairs @FP_FIXED@ false-positive events and introduces @FP_NEW@. These are paired image--seed events, not counts of independent products: each image is evaluated three times. The saved paths and decision arrays support a full audit.

For defects covering at most 0.5\% of the common mask, pooled event-level FNR changes from @TINY_CORE@\% to @TINY_HI@\%. Figure~\ref{fig:size} includes the other sizes and aggregation variants. This event-weighted diagnostic differs from the category-macro metrics in the main table; neither denominator should be silently substituted for the other.
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/miss_by_size.pdf}
\caption{Miss rates by defect area on the common 224 ground truth. Bars pool image--seed events; class composition varies across bins.}\label{fig:size}
\end{figure}
\textbf{Category regressions.} Higher resolution is not uniformly beneficial. On macaroni1, mean FNR rises from 54.33\% to 66.00\%; on macaroni2, from 57.33\% to 65.33\%. Pipe-fryum also worsens from 10.33\% to 13.00\%. The aggregate gain is therefore accompanied by clear category-specific failures. Peak scoring recovers part of these losses, but does not eliminate the weak macaroni performance. No category-specific method switching is fitted to test labels.

\subsection{Aggregation controls and limits}
Using the smoothed-map peak gives FNR @PEAK224_FNR@\% at 224 and @PEAK448_FNR@\% at 448, with FPR @PEAK224_FPR@\% and @PEAK448_FPR@\%. The comparison is exploratory and uses independently normal-calibrated thresholds. Localization scores are exactly inherited because the maps do not change. This controls model and memory differences, while acknowledging that maximum selection and spatial smoothing jointly define the score.

The experiment does not establish an optimal resolution or an optimal score. An exhaustive resolution or weight search against these test outcomes would weaken the interpretation. A separate validation design would be needed to make such choices and obtain a fresh estimate of final performance.
\begin{table}[h]\centering\scriptsize
\caption{Per-category means over three seeds. All columns use normal-only calibration; peak denotes 448 peak. Rates, not percentages.}
\begin{tabular}{lrrrr}\toprule
Category & 224 FNR & 448 FNR & Peak FNR & Peak FPR\\\midrule
@CLASS_TABLE@
\bottomrule\end{tabular}
\end{table}

\clearpage
\twocolumn[{
\section{Qualitative Evidence and Error Review}
\begin{center}\includegraphics[width=.94\textwidth,height=.48\textheight,keepaspectratio]{figures/lowmiss_examples.pdf}

\begin{minipage}{.94\textwidth}\small Paired seed-0 examples for two original and two additional categories. Within each category the rule selects the first recovered miss by filename, otherwise the first introduced miss, otherwise the first anomalous image; records state which rule applied. These outcome-conditioned examples illustrate behavior, not prevalence. Columns: input, GT, calibrated 224/448 heatmaps, and a GT-centered crop. Maps share a range after division by their own pixel thresholds.\end{minipage}
\end{center}
}]
The original failure review contains 24 visually inspected full-memory errors: ten false positives and fourteen false negatives across the six required classes. Selection attempts two of each per class and fills shortages with other errors; cashew contributes four false negatives. It is an illustrative review rather than a random error sample. A further twelve Coreset448 errors (one FP and one FN per additional category) were inspected, bringing the reviewed set to 36. Thin PCB pins remain weakly represented in several misses; normal board borders and component tips produce false alarms. These observations suggest a future masking audit, not a demonstrated cause. File paths, scores, thresholds and individual observations are retained in the artifacts.

\textbf{Normal boundaries and arrangement.} Intact candle rims, normal fryum spokes, different capsule orientations and image boundaries can produce large responses. These observations motivate reference coverage and geometry checks, but do not prove that compression caused the error: many existed with full memory.

\textbf{Small and weak defects.} Several capsule, cashew and chewinggum misses have localized responses below the image threshold. The new resolution and aggregation ablations turn that observation into a controlled test. A visible hotspot is not itself a correct image-level decision, and the plotted ground-truth crop is an analysis aid, never an input to the detector.

\textbf{Background and texture.} Some false alarms respond above an object or near the field boundary; pipe-fryum errors include normal texture and weak boundary blemishes. The upstream PCA mask is not guaranteed to remove all irrelevant background. We keep it unchanged here rather than mixing a segmentation modification into the resolution comparison.

\textbf{Interpretation.} Higher resolution changes representation and query matching cost while keeping stored-bank bytes fixed. Recovered examples support inspectability, while the complete paired error counts establish the measured balance. Neither selected images nor an aggregate gain demonstrate universal improvement. The additional PCB cases broaden the visual evidence but remain part of the same benchmark and development process.
\clearpage
\section{Selective Review, Resources, and Reproduction}
\subsection{Deferral is not detection}
We retain the hard NG threshold and add lower normal-calibration quantiles 0.80, 0.85, 0.90 and 0.95. A score above the lower threshold but not above NG is REVIEW; a lower score is automatically OK. Quantiles were fixed before resolution outcomes. We report review load, automatic coverage, and anomalies incorrectly passed automatically, without assuming any human-review accuracy.

For the predefined 0.90 point with Coreset448, review load is @REVIEW_LOAD@\%, automatic coverage @COVERAGE@\%, and @ESCAPED@\% of anomalies are incorrectly auto-passed. This latter quantity is not ordinary binary FNR: anomalies deferred for review are unresolved, not detected. The curve is more informative than declaring a low escaped rate while hiding a large review workload.
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/selective_review.pdf}
\caption{Predefined review settings. Left: fraction of anomalies automatically passed. Right: error rate among automatic decisions. Neither curve assumes reviewers correctly resolve deferred images.}
\end{figure}
\subsection{Computation and interface}
Experiments use an RTX A500 Laptop GPU (4 GB) and four CPU FAISS threads. Cached feature and matching components are not end-to-end latency. The table instead measures a common warmed application path. Equal 6 MiB banks do not equalize query costs: 448 has more patches and fusion invokes the backbone twice.

\begin{table}[h]\centering\scriptsize
\caption{Warmed single-process demo measurements on 24 fixed images per method, with randomized method order. Mean and 95th-percentile wall time include image/bank loading, features and scoring; initialization and rendering are excluded. GPU is maximum allocated MiB, including resident backbone.}
\begin{tabular}{lrrr}\toprule
Method & Mean ms & P95 ms & GPU MiB\\\midrule
@COST_TABLE@
\bottomrule\end{tabular}
\end{table}

The interface defaults to 448 peak after this exploratory comparison, without an independent post-selection test. It accepts a category and image, showing scores, thresholds, OK/NG or optional REVIEW, heatmap, overlay and calibrated mask. Ninety-six fresh predictions cover the first good/bad filenames in twelve categories and four variants; all scores and maps match formal outputs. Original recordings add 24 verified predictions. Mistakes remain visible. This is image inference, not a camera-acquisition recording.
\subsection{Reproducibility and remaining limits}
The artifact contains manifests, source revisions, checkpoint provenance, per-run metrics, a documented fixed protocol, and unit tests for metric ties, coverage selection, scale fusion and review boundaries. Historical metrics are stored under \texttt{reference\_results}, separate from fresh-run completion markers. Full images, weights and per-pixel predictions are excluded from the public archive and regenerated locally.

The 480 main metric records include 72 score-only peak variants that reuse parent maps, not 480 independent trained models. The 54 brightness settings are separate. Three reference draws provide limited uncertainty information. Calibration uses extra normals, the AE is a small fixed-budget baseline, backbone resources differ, and all findings concern one dataset with resized metric masks. Higher-resolution gains do not establish general industrial safety or CVPR-level novelty. The public repository is available; arXiv deposition and course-platform submission remain author steps.
\section{Conclusion}
The project connects a measured failure---missed small defects under normal-only calibration---to a controlled resolution intervention, with equal stored-bank bytes, repeated references, localization evaluation and paired errors. Compression quality, ranking and operating-point reliability answer different questions. The complete comparison and fusion and aggregation outcomes should guide interpretation rather than be replaced by a single favorable number.
\paragraph{Attribution and assistance.} Upstream methods are credited. AI assisted code, experiments and writing; contributors must review the submission and course disclosure requirements.
\clearpage
{\small\bibliographystyle{ieeenat_fullname}\bibliography{references}}
\end{document}
'''
replace={'AUTHOR':re.search(r'(?m)^\\author\{.*$',old).group(0),'FOOTNOTE':next((x for x in old.splitlines() if x.startswith(r'\begingroup\renewcommand\thefootnote')), ''),'CLASS_TABLE':classtable,'COST_TABLE':costtable,'NTEST':str(ntest),'SCOPE_TABLE':scopetable,'FULL_AUC':val('full','image_auroc'),'ONE_AUC':val('full','image_auroc',shots=1),'ECC_AUC':val('template_ecc','image_auroc'),'AE_AUC':val('autoencoder','image_auroc'),'CLIP_AUC':val('anomalyclip','image_auroc',shots=0),'RANDOM_AUC':val('random25','image_auroc'),'POOL_AUC':val('pooled25','image_auroc'),'CORE_AUC':val('coreset25','image_auroc'),'FN_FIXED':str(counts['fn_fixed']),'FN_NEW':str(counts['fn_introduced']),'FP_FIXED':str(counts['fp_fixed']),'FP_NEW':str(counts['fp_introduced']),'TINY_CORE':f"{sizevalues['coreset25']:.2f}",'TINY_HI':f"{sizevalues['hi448']:.2f}",'REVIEW_LOAD':f"{rv['review_fraction']*100:.2f}",'COVERAGE':f"{rv['automatic_coverage']*100:.2f}",'ESCAPED':f"{rv['escaped_anomaly_fraction']*100:.2f}"}
for prefix,method in [('CORE','coreset25'),('HI','hi448'),('FUSE','multiscale'),('PEAK224','core224_peak'),('PEAK448','hi448_peak')]:
 for metric in ['FNR','FPR']:replace[prefix+'_'+metric]=percent(method,'image_'+metric.lower())
for key,value in replace.items():text=text.replace('@'+key+'@',value)
assert not re.search(r'@[A-Z_]+@',text)
(report/'main.tex').write_text(preamble+text,encoding='utf-8')
print('ALL12_REPORT_WRITTEN',ntest)
