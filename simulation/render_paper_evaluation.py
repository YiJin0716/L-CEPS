"""Replace the note's old evaluation with the complete fixed small-sample suite."""
import json,re,shutil,tempfile
from pathlib import Path
import numpy as np
from current_data import ROOT,DATA,sha,write_json
from render_current_note import draw,compile_note,FIGURES

PLANFILE=DATA/'paper_evaluation_protocol.json'
PLOTFILE=DATA/'paper_evaluation_plot_data.json'
NOTEMANIFEST=DATA/'paper_evaluation_note.json'
METHODS=['the_model','gb_ipss','lasso_ipss','lasso_cv']
COMMONPLAN=DATA/'common_benchmark_protocol.json'

def common_groups():
    plan=json.loads(COMMONPLAN.read_text());settings=plan['settings'];groups=[]
    for family,title in [('sparse_linear','Sparse linear response: correlated Gaussian predictors'),
                         ('friedman1','Friedman #1: independent uniform predictors')]:
        groups.append(dict(title=title,cases=[i for i,s in enumerate(settings) if s['family']==family],
            kind='synthetic',section='appendix',origin='common_benchmark'))
    for dataset in ('mnist','basehock'):
        for family in ('linear','square'):
            groups.append(dict(title=f'{dataset.upper()} predictors: simulated {family} response',
                cases=[i for i,s in enumerate(settings) if s.get('dataset')==dataset and s['family']==family],
                kind='semisynthetic',section='appendix',origin='common_benchmark'))
    assert sorted(i for g in groups for i in g['cases'])==list(range(12))
    with np.load(DATA/'common_benchmark_counts.npz') as a:
        for g in groups:
            g.update(targets=plan['targets'],settings=len(g['cases']),values={},datasets=int(np.isin(a['cases'],g['cases']).sum()))
            for m,name in enumerate(METHODS):
                values={'tpr':a['tp'][m]/a['support'][:,None],'fdr':a['fp'][m]/np.maximum(a['tp'][m]+a['fp'][m],1)}
                g['values'][name]={metric:np.mean([v[a['cases']==c].mean(0) for c in g['cases']],axis=0).tolist() for metric,v in values.items()}
    return groups

def combined_groups(plan):
    original=groups_from_counts(plan);extra=common_groups()
    return original[:12]+extra[:2]+original[12:]+extra[2:]

def groups_from_counts(plan):
    settings=plan['settings'];groups=[]
    for i,s in enumerate(settings[:8]):
        ratio='n = p' if s['ratio']==1 else 'n about 0.8p'
        source='Gaussian' if s['suite']=='gaussian' else 'RNA-seq'
        response='regression' if s['response']=='reg' else 'classification'
        groups.append(dict(title=f'IPSS {source} design: {response}, {ratio}',cases=[i],kind='synthetic' if s['suite']=='gaussian' else 'semisynthetic',section='main'))
    for family in ('linear','square','sine','interaction'):
        ids=[i for i,s in enumerate(settings) if s['suite']=='simple' and s['family']==family]
        groups.append(dict(title=f'Simple {family} response: Gaussian predictors',cases=ids,kind='synthetic',section='appendix'))
    for dataset in ('isolet','har','coil20'):
        for family in ('linear','square'):
            ids=[i for i,s in enumerate(settings) if s['suite']=='real' and s['dataset']==dataset and s['family']==family]
            groups.append(dict(title=f'{dataset.upper()} predictors: simulated {family} response',cases=ids,kind='semisynthetic',section='appendix'))
    assert sorted(i for g in groups for i in g['cases'])==list(range(len(settings)))
    with np.load(DATA/'paper_evaluation_counts.npz') as a:
        for g in groups:
            g['targets']=plan['targets'];g['settings']=len(g['cases']);g['values']={}
            g['datasets']=int(np.isin(a['cases'],g['cases']).sum())
            for m,name in enumerate(METHODS):
                values={'tpr':a['tp'][m]/a['support'][:,None],'fdr':a['fp'][m]/np.maximum(a['tp'][m]+a['fp'][m],1)}
                g['values'][name]={metric:np.mean([v[a['cases']==c].mean(0) for c in g['cases']],axis=0).tolist() for metric,v in values.items()}
    return groups

def description(group,plan,sources):
    if group.get('origin')=='common_benchmark':
        plan=json.loads(COMMONPLAN.read_text())
    s=plan['settings'][group['cases'][0]]
    dimension=r'$p=\operatorname{round}(\exp U)$, $U\sim\mathrm{Uniform}(\log50,\log500)$'
    size=r'$n=p$' if s['ratio']==1 else r'$n=\max\{50,\lfloor0.8p\rfloor\}$'
    if s['suite']=='gaussian':
        response=(r'Center the signal, draw SNR uniformly from $[0.5,2]$, and add independent Gaussian noise with variance equal to the signal variance divided by SNR. Finally center the response.' if s['response']=='reg' else r'Center the signal, draw $u\sim\mathrm{Uniform}(1,3)$, and generate independent Bernoulli responses with probabilities $\operatorname{sigmoid}(u s_i)$.')
        return (r'\textbf{Gaussian '+('regression' if s['response']=='reg' else 'classification')+r', '+size+r'.} These 100 datasets use '+dimension+r' and '+size+r'. Draw Gaussian predictors with covariance $\Sigma_{jk}=0.5^{|j-k|}$ and standardize their columns. Choose $k$ uniformly from $\{5,\ldots,15\}$ and choose $k$ relevant columns uniformly. The uncentered signal is $\sum_{j\in S}\exp(-X_{ij}^2)$. '+response)
    if s['suite']=='rnaseq':
        response=(r'Draw SNR uniformly from $[0.5,2]$ and add independent Gaussian noise with variance $\operatorname{var}(s)/\mathrm{SNR}$. Finally center the response.' if s['response']=='reg' else r'Draw $u\sim\mathrm{Uniform}(1,3)$ and sample independent $y_i\sim\mathrm{Bernoulli}(\operatorname{sigmoid}(u s_i))$.')
        return (r'\textbf{RNA-seq '+('regression' if s['response']=='reg' else 'classification')+r', '+size+r'.} These 100 datasets use '+dimension+r' and '+size+r'. Sample rows and varying columns without replacement from the $569\times6426$ ovarian RNA-seq matrix, then standardize each selected column. Choose $k\in\{10,\ldots,30\}$ relevant genes uniformly and split them into $G\in\{\lfloor k/2\rfloor,\ldots,k\}$ random nonempty groups. Standardize each group sum, apply an independently randomized tanh or Gaussian-bump function as specified above, sum the outputs, and center the result $s$. '+response)
    formulas={'linear':r'\sum_{j\in S}X_{ij}','square':r'\sum_{j\in S}X_{ij}^2','sine':r'\sum_{j\in S}\sin(X_{ij})','interaction':r'X_{i,j_1}X_{i,j_2}+X_{i,j_3}X_{i,j_4}+X_{i,j_5}X_{i,j_6}'}
    text=(r' These 200 datasets give equal weight to 100 draws with $n=p$ and 100 with $n=\max\{50,\lfloor0.8p\rfloor\}$; in both, '+dimension+r'. ')
    if s['suite']=='synthetic':
        if s['family']=='sparse_linear':
            return (r'\textbf{Sparse linear response with correlated Gaussian predictors.}'+text+
                r'Draw independent rows $X_i\sim N(0,\Sigma)$ with $\Sigma_{jk}=0.25^{|j-k|}$. Choose ten relevant columns uniformly and set '
                r'$y_i=(4.5/\sqrt n)\sum_{j\in S}X_{ij}+\varepsilon_i$, with independent $\varepsilon_i\sim N(0,1)$. '
                r'This is the sparse linear benchmark family used in the model-X knockoffs authors\textquotesingle{} example \cite{knockoffexample}. '
                r'We retain its correlation, coefficient scale and noise variance, but use ten relevant columns instead of sixty to accommodate $p\ge50$, and replace its fixed dimensions by the rule above. '
                r'The generator does not sample-standardize $X$ or center $y$. All draws are retained.')
        return (r'\textbf{Friedman \#1 nonlinear regression.}'+text+
            r'Draw all entries of $X$ independently from $U(0,1)$. Before permuting columns, set '
            r'$y_i=10\sin(\pi X_{i1}X_{i2})+20(X_{i3}-0.5)^2+10X_{i4}+5X_{i5}+\varepsilon_i$, '
            r'with independent $\varepsilon_i\sim N(0,1)$. Only these five columns are relevant. '
            r'Use the standard Friedman \#1 generator \cite{friedman1}, then randomly permute all columns and track the five relevant indices. '
            r'The generator does not standardize $X$ or center $y$. All draws are retained.')
    if s['suite']=='simple':
        text=r'\textbf{Simple '+s['family']+r' response.}'+text+r'Generate Gaussian predictors with $\Sigma_{jk}=0.5^{|j-k|}$ and standardize their columns. '
    else:
        source=sources[s['dataset']];n,p=source['shape']
        detail={'isolet':r'The pool is the scikit-feature subset of spoken-letter acoustic measurements \cite{scikitfeature}.',
                'har':r'The pool combines the original UCI train and test predictor files of smartphone sensor summaries \cite{har}.',
                'coil20':r'The pool contains flattened object images from COIL20, distributed by scikit-feature \cite{scikitfeature}.',
                'mnist':r'The pool is the official MNIST training set, with each $28\times28$ grayscale image flattened to 784 pixel measurements \cite{mnist}.',
                'basehock':r'The pool is the BASEHOCK text feature matrix from scikit-feature, derived from two 20 Newsgroups categories \cite{scikitfeature}.'}[s['dataset']]
        text=r'\textbf{'+s['dataset'].upper()+' predictors with a '+s['family']+r' response.}'+text+f'The fixed real predictor pool has {n:,} rows and {p:,} columns. '+detail+r' Sample rows without replacement, remove columns constant in those rows, sample $p$ remaining columns without replacement, and standardize them. Original response labels are unused. '
    text+=r'Choose six relevant columns uniformly. Form the signal $v_i='+formulas[s['family']]+r'$, center it and scale it to standard deviation one, and set $y_i=v_i+\varepsilon_i$ with independent $\varepsilon_i\sim N(0,1)$. Thus SNR is 1. Finally center the response. All draws are retained.'
    return text

def figure(group,index,plan,sources):
    return ('\n'+r'\begin{figure}[H]'+'\n'+description(group,plan,sources)+'\n'+r'\par\medskip\centering'+'\n'+
        r'\includegraphics[page='+str(index)+r',width=\linewidth]{final_model_note_figures.pdf}'+'\n'+
        r'\caption{'+('Small-sample adaptation of the IPSS '+('Gaussian' if index<=4 else 'RNA-seq')+' design.' if index<=8 else 'Prespecified supplemental setting; both sample-size regimes receive equal weight.')+r'}'+'\n'+
        r'\label{fig:assessment-'+str(index)+'}'+'\n'+r'\end{figure}'+'\n')

def additional_results(groups):
    extra=[(i,g) for i,g in enumerate(groups,1) if g.get('origin')=='common_benchmark']
    assert len(extra)==6 and sum(g['datasets'] for _,g in extra)==1200
    def percent(group,method,metric):
        index=int(np.flatnonzero(np.isclose(group['targets'],.1))[0])
        return f"{100*group['values'][method][metric][index]:.1f}"+r'\%'
    def pair(group,method):
        return percent(group,method,'tpr')+' TPR and '+percent(group,method,'fdr')+' FDR'
    def reference(item):
        return r'Figure~\ref{fig:assessment-'+str(item[0])+'}'
    linear,friedman,mnist_linear,mnist_square,base_linear,base_square=extra
    paragraphs=[r'''
\subsection{Results on the additional benchmarks}
The two additional synthetic designs and the MNIST and BASEHOCK
semi-simulations contribute 1,200 datasets across 12 settings. Each of the
six figures averages 100 draws with $n=p$ and 100 draws with
$n=\max\{50,\lfloor0.8p\rfloor\}$. The following numerical comparisons
use target FDR $\alpha=0.10$; the appendix shows the full target range
from 0 to 0.50 for every method and setting.
''',
        r'\textbf{Synthetic benchmarks.} In the sparse linear design ('+reference(linear)+'), the model achieves '+pair(linear[1],'the_model')+
        ', compared with '+pair(linear[1],'gb_ipss')+' for GB-IPSS and '+pair(linear[1],'lasso_ipss')+
        r' for Lasso-IPSS. The higher TPR comes with empirical FDR above the 10\% target. In Friedman \#1 ('+reference(friedman)+'), the model achieves '+
        pair(friedman[1],'the_model')+', whereas GB-IPSS achieves '+pair(friedman[1],'gb_ipss')+' and Lasso-IPSS achieves '+pair(friedman[1],'lasso_ipss')+
        '. GB-IPSS therefore finds more true variables in this design while also remaining below the stated FDR target.',
        r'\textbf{Real predictors with linear responses.} On MNIST ('+reference(mnist_linear)+'), the model achieves '+pair(mnist_linear[1],'the_model')+
        ', compared with '+pair(mnist_linear[1],'gb_ipss')+' for GB-IPSS and '+pair(mnist_linear[1],'lasso_ipss')+
        ' for Lasso-IPSS. On BASEHOCK ('+reference(base_linear)+'), the corresponding results are '+pair(base_linear[1],'the_model')+
        ' for the model, '+pair(base_linear[1],'gb_ipss')+' for GB-IPSS, and '+pair(base_linear[1],'lasso_ipss')+
        '. In both designs, the model finds more true variables than Lasso-IPSS and has lower empirical FDR than GB-IPSS.',
        r'\textbf{Real predictors with square responses.} On MNIST ('+reference(mnist_square)+'), the model achieves '+pair(mnist_square[1],'the_model')+
        ', compared with '+pair(mnist_square[1],'gb_ipss')+' for GB-IPSS and '+pair(mnist_square[1],'lasso_ipss')+
        ' for Lasso-IPSS. On BASEHOCK ('+reference(base_square)+'), the model achieves '+pair(base_square[1],'the_model')+
        ', compared with '+pair(base_square[1],'gb_ipss')+' for GB-IPSS and '+pair(base_square[1],'lasso_ipss')+
        r' for Lasso-IPSS. The model selects fewer than one in ten true variables in both square-response designs at $\alpha=0.10$, so substantial power limitations remain.',
    ]
    cv_tpr=[g['values']['lasso_cv']['tpr'][0] for _,g in extra]
    cv_fdr=[g['values']['lasso_cv']['fdr'][0] for _,g in extra]
    paragraphs.append(r'\textbf{LassoCV.} Across these six comparisons, LassoCV has TPR between '+f'{100*min(cv_tpr):.1f}'+r'\% and '+f'{100*max(cv_tpr):.1f}'+
        r'\%, with FDR between '+f'{100*min(cv_fdr):.1f}'+r'\% and '+f'{100*max(cv_fdr):.1f}'+
        r'\%. Its large selected sets recover many true variables but also include many false selections; LassoCV has no target-FDR input.')
    return '\n\n'.join(paragraphs)+'\n'

def build_source(groups,plan):
    path=ROOT/'final_model_note.tex';prefix=path.read_text().split(r'\section{Evaluation}')[0]
    prefix=prefix.replace('the easier and harder appendix difficulties','two additional validation difficulties')
    prefix=prefix.replace('The model is then frozen before generating the 5,400 confirmation datasets\ndescribed below. No training or model selection uses their outcomes.',
        'The model remains frozen for the new evaluation below. No training or\nmodel selection uses these new evaluation outcomes.')
    sources=json.loads((DATA/'paper_evaluation_sources.json').read_text())
    sources.update(json.loads((DATA/'common_benchmark_sources.json').read_text()))
    evaluation=r'''
\section{Evaluation}\label{sec:experiments}
The evaluation focuses on limited sample size relative to the number of variables.
For every dataset, draw
\[
 U\sim\mathrm{Uniform}(\log50,\log500),\qquad
 p=\operatorname{round}(\exp U),
\]
and use either $n=p$ or $n=\max\{50,\lfloor0.8p\rfloor\}$.
Thus $50\le n\le p\le500$. There are 40 prespecified combinations of
data source, response rule and sample-size regime, with 100 new datasets
per combination, for 4,000 datasets in total. The original 28 combinations
contribute 2,800 datasets, and a subsequent addition of 12 combinations
contributes 1,200 datasets. Each batch's dimensions, random seeds,
generators and model files were fixed before measuring that batch's performance.
All settings and all repetitions are reported; none are selected for a
favorable result. The model is not refitted for this evaluation.

The first eight figures use Gaussian and RNA-seq designs based on
\cite{ipssgb}, with the dimensions above and the data generation specified
below. The appendix adds
simple synthetic responses, two standard synthetic benchmark families,
and five additional real predictor sources. The two added synthetic
families are sparse correlated-Gaussian linear regression and Friedman \#1;
the two added real sources are MNIST and BASEHOCK.
Some predictor distributions and response functions overlap fitting.
The study measures small-sample selection performance, rather than
assuming that every response mechanism is new to the model.

\subsection{Comparisons and averaging}
Compare the model with GB-IPSS, Lasso-IPSS and LassoCV. Every method
receives the same $X,y$ within a draw. The two IPSS baselines share their
stability paths with the model. GB-IPSS uses $\delta=1.25$ for continuous
responses and $\delta=1$ for binary responses, matching Table S1 of
\cite{ipssgb}. Lasso-IPSS uses its default correlation rule: setting
self-correlations to zero, let $a$ be the mean absolute correlation over
all $p^2$ entries and $b$ the mean columnwise maximum. Its $\delta$ is 1
if $a\le0.05$, 0 if $a\ge0.25$, and
$\operatorname{clip}(2.25-5a-2b,0,1)$ otherwise. LassoCV selects the
nonzero coefficients of the model's linear LassoCV measurement, with
five-fold prediction-error selection; it also treats binary $y$ as numeric
0/1 and has no target-FDR input. These comparisons use the retained
implementation of these methods, not all methods from the original paper.

Targets are $\alpha=0,0.01,\ldots,0.50$. Within each setting, average
realized FDP and TPR over the 100 datasets. Where a figure combines the
two sample-size regimes, give each regime equal weight. The left panel
shows empirical FDR and the right panel shows TPR, with the full target
range displayed. Real-predictor experiments repeatedly sample fixed
observed matrices and generate new responses; they are not independent
collections of new patients, speakers, subjects or objects. Their truth
is the planted response support. Predictive proxies are counted as false
selections when they are outside that support. No general FDR guarantee
is asserted for the learned selection rule.

The curves are Monte Carlo means; per-setting standard errors are retained
with the numeric results. All 4,000 inputs are regenerated and checked
against their saved hashes, and TP/FP counts are recounted from the saved
selected sets. The model's selection rule is independently reconstructed
for every target and dataset. The public application interface is also
checked on one dataset per setting using the saved IPSS paths. The data
generators and frozen draws are retained in \code{paper_evaluation.py},
\code{common_benchmarks.py} and their respective protocol files.

\subsection{Gaussian design adapted from IPSS}
For each dataset, draw $n$ independent Gaussian rows with mean zero and
covariance $\Sigma_{jk}=0.5^{|j-k|}$, then standardize each column to mean
zero and variance one. Draw $k$ uniformly from $\{5,\ldots,15\}$ and choose
a set $S$ of $k$ relevant columns uniformly without replacement. Calculate
the signal and subtract its sample mean:
\[
 r_i=\sum_{j\in S}\exp(-X_{ij}^2),\qquad
 s_i=r_i-\frac1n\sum_{\ell=1}^n r_\ell.
\]
For regression, draw one SNR uniformly from $[0.5,2]$ per dataset and set
$\sigma^2=(n^{-1}\sum_i s_i^2)/\mathrm{SNR}$. Draw independent
$\varepsilon_i\sim N(0,\sigma^2)$, form $\widetilde y_i=s_i+\varepsilon_i$,
and subtract the sample mean of $\widetilde y$ to obtain $y$.
For classification, draw one $u\sim\mathrm{Uniform}(1,3)$ per dataset and
generate the labels independently, conditional on the predictor matrix
and $u$, using
\[
 \pi_i=\frac{1}{1+\exp(-u s_i)},\qquad
 y_i\sim\mathrm{Bernoulli}(\pi_i).
\]
Sample variances in these data-generation steps use divisor $n$.
'''
    evaluation+=''.join(figure(g,i,plan,sources) for i,g in enumerate(groups[:4],1))
    evaluation+=r'''
\subsection{RNA-seq design adapted from IPSS}
The source is the ovarian RNA-seq matrix released with the paper:
569 patients and 6,426 measured genes. Sample $n$ rows without replacement,
exclude columns constant in those rows, and sample $p$ remaining columns
without replacement. Standardize each selected column to mean zero and
variance one. Draw $k$ uniformly from $\{10,\ldots,30\}$ and choose $k$
relevant columns uniformly without replacement.

Draw the number of groups $G$ uniformly from $\lfloor k/2\rfloor$ through
$k$. Randomly permute the $k$ relevant columns and uniformly choose $G-1$
cut positions among the $k-1$ internal positions, giving $G$ nonempty groups.
For each group $S_g$, sum its columns and standardize the sum across samples
to obtain $z_g$. Independently draw a function for each group:
\[
 f(z)=\begin{cases}
 d_1[1+\tanh(a(d_2z-b))]/2,&\text{with probability }1/2,\\
 d_1\exp(-c z^2),&\text{with probability }1/2,
 \end{cases}
\]
where $a\sim U(0.5,1.5)$, $b\sim U(-1,1)$, $c\sim U(1,3)$ and
$d_1,d_2$ are independent equiprobable signs. The parameters and the choice
between the two functions are drawn once per group and shared across its
samples. Sum the group contributions and center the signal:
\[
 r_i=\sum_{g=1}^G f_g(z_{ig}),\qquad
 s_i=r_i-\frac1n\sum_{\ell=1}^n r_\ell.
\]
Generate responses from $s$ using the same rules as in the Gaussian design:
for regression, draw one SNR from $U(0.5,2)$, add independent Gaussian noise
with variance $(n^{-1}\sum_i s_i^2)/\mathrm{SNR}$, and center the response;
for classification, draw one $u\sim U(1,3)$ and independently sample
$y_i\sim\mathrm{Bernoulli}([1+\exp(-u s_i)]^{-1})$ conditional on the
predictor matrix, the generated signal functions and $u$.
'''
    evaluation+=''.join(figure(g,i,plan,sources) for i,g in enumerate(groups[4:8],5))
    evaluation+=additional_results(groups)
    references=r'''
\begingroup\small
\begin{thebibliography}{9}
\bibitem{ipss} O.~Melikechi and J.~W.~Miller.
\emph{Integrated path stability selection}. arXiv:2403.15877, 2025.
\bibitem{ipssgb} O.~Melikechi, D.~B.~Dunson, and J.~W.~Miller.
\emph{Nonparametric IPSS: Fast, flexible feature selection with false discovery control}.
Bioinformatics, 2025. doi:10.1093/bioinformatics/btaf299.
The specification also uses the supplement in arXiv:2410.02208v3.
Code and RNA-seq data: \url{https://github.com/omelikechi/ipss_bioinformatics}.
\bibitem{xgboost} T.~Chen and C.~Guestrin.
\emph{XGBoost: A Scalable Tree Boosting System}. KDD, 2016, pp.~785--794.
\bibitem{scikitfeature} J.~Li et al. scikit-feature dataset collection.
ISOLET subset, COIL20 and BASEHOCK matrices, repository snapshot
\code{48cffad4e88ff4b9d2f1c7baffb314d1b3303792}.
\url{https://jundongl.github.io/scikit-feature/datasets}.
\bibitem{har} D.~Anguita et al. Human Activity Recognition Using Smartphones.
UCI Machine Learning Repository. doi:10.24432/C54S4K.
\bibitem{knockoffexample} M.~Sesia. \emph{Controlled variable Selection with Model-X Knockoffs}.
Authors' software example, 2017.
\url{https://web.stanford.edu/group/candes/knockoffs/software/knockoffs/r/knockoff.html}.
\bibitem{friedman1} J.~H.~Friedman. \emph{Multivariate adaptive regression splines}.
Annals of Statistics, 19(1):1--67, 1991. Generator specification:
\url{https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html}.
\bibitem{mnist} TensorFlow/Keras. MNIST handwritten-digit dataset.
Official training split: 60,000 images, each $28\times28$ pixels.
\url{https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data}.
\end{thebibliography}
\endgroup
\newpage\appendix\section{Additional Small-Sample Evaluations}
\subsection{Simulated test datasets}
'''
    appendix='';real_started=False
    for i,g in enumerate(groups[8:],9):
        if g['kind']=='semisynthetic' and not real_started:
            appendix+=r'\subsection{Real predictors with simulated responses}'+'\n';real_started=True
        appendix+=figure(g,i,plan,sources)
    path.write_text(prefix+evaluation+references+appendix+'\n'+r'\end{document}'+'\n')

def render():
    manifest=json.loads(NOTEMANIFEST.read_text());plan=json.loads(PLANFILE.read_text())
    assert manifest['protocol_sha256']==sha(PLANFILE)
    assert manifest['counts_sha256']==sha(DATA/'paper_evaluation_counts.npz')
    assert manifest['common_protocol_sha256']==sha(COMMONPLAN)
    assert manifest['common_counts_sha256']==sha(DATA/'common_benchmark_counts.npz')
    groups=json.loads(PLOTFILE.read_text());expected=combined_groups(plan)
    assert groups==expected
    assert [g for g in groups if g.get('origin')!='common_benchmark']==groups_from_counts(plan)
    draw(groups);checks=compile_note()
    assert len(checks['actual_pdf_figure_checks'])==24
    write_json(DATA/'current_note_verification.json',dict(assessment='paper_small_sample',model_kind='mlp_screened_single_f',
        source_sha256=sha(__file__),renderer_sha256=sha(Path(__file__).with_name('render_current_note.py')),
        note_source_sha256=sha(ROOT/'final_model_note.tex'),pdf_sha256=sha(ROOT/'final_model_note.pdf'),
        figures_sha256=sha(ROOT/FIGURES),active_manifest_sha256=sha(ROOT/'models/active_model.json'),
        weights_sha256=sha(ROOT/'models/retained_mlp_hard_rows.pt'),protocol_sha256=sha(PLANFILE),
        plot_data_sha256=sha(PLOTFILE),plot_counts_sha256=manifest['counts_sha256'],
        assessment_audit_sha256=sha(DATA/'paper_evaluation_audit.json'),
        public_api_verification_sha256=sha(DATA/'paper_evaluation_verification.json'),
        common_protocol_sha256=sha(COMMONPLAN),common_counts_sha256=manifest['common_counts_sha256'],
        common_audit_sha256=sha(DATA/'common_benchmark_audit.json'),
        original_18_plot_data_groups_unchanged=True,
        all_plotted_curves_recounted=True,all_settings_reported=True,datasets=4000,settings=40,
        paired_figures=24,main_figures=8,appendix_simulated_figures=6,appendix_real_predictor_figures=10,
        appendix_only_figures_and_dataset_introductions=True,appendix_selected_after_evaluation=False,
        displayed_methods=METHODS,**checks))
    print('Rebuilt small-sample note:',checks['pages'],'pages, 24 paired figures.',flush=True)

def install():
    plan=json.loads(PLANFILE.read_text());verification=json.loads((DATA/'paper_evaluation_verification.json').read_text())
    audit=json.loads((DATA/'paper_evaluation_audit.json').read_text())
    assert verification['model_files_unchanged'] and verification['all_dimensions_follow_protocol']
    assert audit['protocol_sha256']==sha(PLANFILE) and audit['counts_sha256']==sha(DATA/'paper_evaluation_counts.npz')
    assert audit['datasets_regenerated']==2800 and audit['all_model_selections_independently_reconstructed']
    common_audit=json.loads((DATA/'common_benchmark_audit.json').read_text())
    assert common_audit['protocol_sha256']==sha(COMMONPLAN)
    assert common_audit['counts_sha256']==sha(DATA/'common_benchmark_counts.npz')
    assert common_audit['datasets_regenerated']==1200 and common_audit['frozen_files_unchanged']
    assert common_audit['public_api_cases']==list(range(12))
    groups=combined_groups(plan)
    backup=Path(tempfile.mkdtemp(prefix='ipss-before-paper-evaluation-'))
    for path in [ROOT/'final_model_note.tex',ROOT/'final_model_note.pdf',ROOT/FIGURES,DATA/'current_note_verification.json']:
        shutil.copy2(path,backup/path.name)
    write_json(PLOTFILE,groups);build_source(groups,plan)
    write_json(NOTEMANIFEST,dict(kind='paper_small_sample',protocol_sha256=sha(PLANFILE),counts_sha256=sha(DATA/'paper_evaluation_counts.npz'),
        common_protocol_sha256=sha(COMMONPLAN),common_counts_sha256=sha(DATA/'common_benchmark_counts.npz'),
        all_settings_reported=True,previous_note_backup=str(backup)))
    render()

if __name__=='__main__':install()
