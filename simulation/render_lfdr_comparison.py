"""Append all local-FDR comparisons to the existing note; preserve earlier curves."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from current_data import DATA,ROOT,sha,write_json
import lfdr_evaluation as assessment
import render_paper_evaluation as original
from render_current_note import compile_note

FIGURES=ROOT/'lfdr_comparison_figures.pdf'
PLOTDATA=DATA/'lfdr_comparison_plot_data.json'
LABELS={'the_model':'The model','adaptz':'Sun-Cai (marginal z)',
        'cumulative_ell':'Cumulative ell (marginal z)',
        'lasso_mfdr_pointwise':'Lasso local FDR (pointwise)',
        'lasso_mfdr_cumulative':'Lasso local FDR (cumulative)'}
COLORS={'the_model':'#245b8d','adaptz':'#555555','cumulative_ell':'#3a8f66',
        'lasso_mfdr_pointwise':'#8c61aa','lasso_mfdr_cumulative':'#d17a20'}
BEGIN='% BEGIN LOCAL FDR COMPARISON'
END='% END LOCAL FDR COMPARISON'


def groups_from_counts():
    source_plan=json.loads(original.PLANFILE.read_text())
    old_groups=original.combined_groups(source_plan)
    assert old_groups==json.loads(original.PLOTFILE.read_text())
    groups=[dict(title='All existing assessment settings',cases=list(range(40)),
                 kind='all',settings=40,datasets=4000)]
    ordered=sorted(enumerate(old_groups),key=lambda item:(item[1]['kind']!='synthetic',item[0]))
    for old_index,source in ordered:
        group=dict(source)
        group['source_cases']=source['cases']
        group['cases']=[i+(28 if source.get('origin')=='common_benchmark' else 0) for i in source['cases']]
        group['original_group_index']=old_index
        groups.append(group)
    assert sorted(i for g in groups[1:] for i in g['cases'])==list(range(40))
    with np.load(assessment.COUNTS) as a:
        assert list(a['methods'])==assessment.METHODS
        for group in groups:
            mask=np.isin(a['cases'],group['cases']);group['datasets']=int(mask.sum())
            group['targets']=a['targets'].tolist();group['values']={}
            for m,name in enumerate(assessment.METHODS):
                tpr=a['tp'][m,mask]/a['support'][mask,None]
                fdr=a['fp'][m,mask]/np.maximum(1,a['tp'][m,mask]+a['fp'][m,mask])
                group['values'][name]={'tpr':tpr.mean(0).tolist(),'fdr':fdr.mean(0).tolist()}
    return groups,old_groups


def draw(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(FIGURES) as pdf:
        for index,group in enumerate(groups):
            fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
            for name,values in group['values'].items():
                for ax,metric in zip(axes,['fdr','tpr']):
                    ax.plot(group['targets'],values[metric],label=LABELS[name],color=COLORS[name],
                            linewidth=2.5 if name=='the_model' else 1.7,
                            linestyle='--' if name=='lasso_mfdr_pointwise' else '-')
            axes[0].plot([0,.5],[0,.5],'k--',linewidth=1,label='Target')
            axes[0].legend(fontsize=7.1,loc='best')
            for ax,label in zip(axes,['False discovery rate','True positive rate']):
                ax.set(xlabel='Target FDR',ylabel=label,xlim=(0,.5),ylim=(0,1));ax.grid(alpha=.2)
            suffix=' (1 setting)' if group['settings']==1 else f" ({group['settings']} equally weighted settings)"
            fig.suptitle(group['title']+suffix)
            fig.tight_layout(rect=(0,0,1,.94));pdf.savefig(fig)
            if index==0:fig.savefig(ROOT/'lfdr_comparison.png',dpi=180)
            plt.close(fig)


def percent(x):return f'{100*x:.1f}'+r'\%'


def comparison_text(result):
    target_index=int(np.flatnonzero(np.isclose(result['targets'],.1))[0])
    index=target_index
    text=r'''
\subsection{Comparison with local false-discovery methods}
The following comparison reuses all 4,000 datasets above, including the
same predictor matrices, responses and planted relevant columns.
The model's saved selections are reused without refitting or changing its
thresholds. The additional methods receive only $X$ and $y$.
Their implementation choices were fixed before computing their test results.

\textbf{Sun--Cai.} For each column, calculate the correlation with $y$,
convert it to a $t$ statistic with $n-2$ degrees of freedom, and transform
its null cumulative probability to a standard normal score $z_j$.
For binary responses this is also the pooled two-sample $t$ test on that
column across the two response groups. Estimate
$\ell_j=\widehat\pi_0\phi(z_j)/\widehat f(z_j)$, truncated to $[0,1]$,
using the theoretical $N(0,1)$ null. The null proportion uses the
Jin--Cai estimator \cite{jincai}; the density uses a Gaussian kernel with
bandwidth chosen by unbiased cross-validation. Implement the Sun--Cai
selection rule \cite{suncai}: sort these scores and take the largest
complete-tie prefix whose average is at most $\alpha$.
This is an implementation of the published rule with the numerical choices
recorded in \code{data/lfdr_comparison_protocol.json}.

\textbf{Cumulative $\ell$-value.} Use the same $z_j$ values, but replace
the estimated density by the mixture in Abraham, Castillo and Roquain
\cite{cumulativeell}. Its null density is standard normal, and its
alternative density is
$g(z)=(1-e^{-z^2/2})/(\sqrt{2\pi}z^2)$, with its continuous value at zero.
Estimate the nonnull mixture weight by maximum likelihood over $[1/p,1]$.
Calculate each variable's posterior null probability and apply the same
cumulative-average rule. The paper studies independent Gaussian
coordinates; using regression-derived $z$ scores here is an application
to the present datasets, not a reproduction of that paper's simulation.

\textbf{Lasso local FDR.} Use the authors' \code{ncvreg} implementation
of Miller and Breheny \cite{millerbreheny}. Fit a Lasso with ten-fold
cross-validation and use the penalty with minimum validation error.
The response family is Gaussian for regression and binomial for
classification. Compute \code{local_mfdr(method="ashr")} for all columns.
The \emph{pointwise} curve selects columns with local FDR at most $\alpha$,
as in the paper's simulation. The \emph{cumulative} curve instead applies
the cumulative-average rule to those same scores. The latter is explicitly
an additional version of the selection rule. Package defaults are retained
otherwise; software versions and random seeds accompany the results.

\textbf{What is being compared.} Every method is scored against the
same columns that directly generated $y$. A correlated but unplanted
column therefore counts as a false selection. This differs from the
marginal-association null used by these literature methods.
Nonlinear responses can also contain relevant columns with zero marginal
linear association. The comparison measures their usefulness for the
present variable-selection task; their original FDR guarantees do not
automatically apply to these correlated, nonlinear datasets.

\textbf{Results.} Table~\ref{tab:lfdr-comparison} reports both error rate
and power at target $\alpha=0.10$, averaged equally across the 40 settings.
Standard errors reflect independent replicates within each fixed setting.
The paired curves cover all targets from 0 to 0.50; the appendix includes
every setting, ordered by simulated predictors and then real predictors.
No setting was removed based on its results.
\begin{table}[H]
\centering\small
\begin{tabular}{lrrrr}
\toprule
Method & TPR & TPR SE & FDR & FDR SE\\
\midrule
'''
    table_names={'the_model':'The model','adaptz':'Sun--Cai', 'cumulative_ell':r'Cumulative $\ell$-value',
                 'lasso_mfdr_pointwise':'Lasso local FDR: pointwise','lasso_mfdr_cumulative':'Lasso local FDR: cumulative'}
    for method in assessment.METHODS:
        d=result['overall'][method]
        text+=table_names[method]+' & '+' & '.join(percent(d[m][index]) for m in ['tpr','tpr_se','fdr','fdr_se'])+r'\\'+'\n'
    text+=r'''\bottomrule
\end{tabular}
\caption{All 4,000 existing datasets, target FDR $0.10$. All entries are percentages; SE denotes standard error.}
\label{tab:lfdr-comparison}
\end{table}
'''
    ours=result['overall']['the_model']
    text+='The model has '+percent(ours['tpr'][index])+' TPR and '+percent(ours['fdr'][index])+r' FDR at this target. '
    for method in assessment.METHODS[1:]:
        d=result['overall'][method];delta=100*d['model_minus_tpr'][index];se=100*d['model_minus_tpr_se'][index]
        text+='Relative to '+table_names[method]+f', its TPR difference is {delta:+.1f} percentage points '+r'(paired 95\% normal interval '+f'{delta-1.96*se:+.1f} to {delta+1.96*se:+.1f}'+'). '
    text+='These power differences must be interpreted together with the measured FDRs in the table.\n'
    violations={method:sum(s['methods'][method]['fdr'][index]>.1 for s in result['settings'])
                for method in assessment.METHODS}
    text+='\nAt this target, the model is the only displayed method with overall empirical FDR below 10'+r'\%. '
    text+=f"Its empirical FDR exceeds 10\\% in {violations['the_model']} of the 40 individual settings; "
    text+='the corresponding counts are '+', '.join(str(violations[m])+' for '+table_names[m] for m in assessment.METHODS[1:])+'. '
    text+='An acceptable overall average therefore does not establish control in every setting.\n'
    def subset_pair(method,synthetic):
        selected=[s for s in result['settings'] if (s['spec']['suite'] in ['gaussian','simple','synthetic'])==synthetic]
        return (percent(np.mean([s['methods'][method]['tpr'][index] for s in selected]))+' TPR and '+
                percent(np.mean([s['methods'][method]['fdr'][index] for s in selected]))+' FDR')
    text+='\nFor the 1,600 datasets with simulated predictors, the model achieves '+subset_pair('the_model',True)+', compared with '+subset_pair('lasso_mfdr_cumulative',True)+' for the cumulative Lasso rule. '
    text+='For the 2,400 datasets with real predictors, the model achieves '+subset_pair('the_model',False)+', compared with '+subset_pair('lasso_mfdr_cumulative',False)+' for that rule. '
    text+='On real predictors, several alternatives recover more planted variables while also selecting many correlated, unplanted columns. '
    text+='The complete figures expose this tradeoff rather than ranking methods by TPR alone.\n'
    text+='\n'+r'\begin{figure}[H]'+'\n'
    text+=r'\textbf{All existing assessment settings.} These 4,000 datasets comprise 40 fixed settings with 100 independent draws each: 1,600 datasets use simulated predictors and 2,400 use real predictors with simulated responses. Each draw satisfies $50\le n\le p\le500$. The same matrices, responses, planted supports and sample-size regimes are used for all methods. The preceding sections specify each generator; all settings receive equal weight.'+'\n'
    text+=r'\par\medskip\centering'+'\n'+r'\includegraphics[page=1,width=\linewidth]{lfdr_comparison_figures.pdf}'+'\n'
    text+=r'\caption{The model and local-FDR methods on all existing assessment settings.}'+'\n'+r'\label{fig:lfdr-overall}'+'\n'+r'\end{figure}'+'\n'
    return text


def build_source(groups,old_groups,result):
    source=(ROOT/'final_model_note.tex').read_text()
    source=re.sub(re.escape(BEGIN)+r'.*?'+re.escape(END)+r'\n?', '', source, flags=re.S)
    block=BEGIN+'\n'+comparison_text(result)+END+'\n'
    marker='\\begingroup\\small\n\\begin{thebibliography}'
    assert marker in source
    source=source.replace(marker,block+'\n'+marker,1)
    bibliography=r'''
\bibitem{suncai} W.~Sun and T.~T.~Cai. Oracle and adaptive compound decision rules for false discovery rate control.
Journal of the American Statistical Association, 102:901--912, 2007. doi:10.1198/016214507000000545.
\bibitem{jincai} J.~Jin and T.~T.~Cai. Estimating the null and the proportion of nonnull effects in large-scale multiple comparisons.
Journal of the American Statistical Association, 102:495--506, 2007. doi:10.1198/016214507000000167.
\bibitem{cumulativeell} K.~Abraham, I.~Castillo, and E.~Roquain. Empirical Bayes cumulative $\ell$-value multiple testing procedure for sparse sequences.
arXiv:2102.00929v3, 2022.
\bibitem{millerbreheny} R.~Miller and P.~Breheny. Feature-specific inference for penalized regression using local false discovery rates.
Statistics in Medicine, 42:1412--1429, 2023. doi:10.1002/sim.9678.
'''
    source=source.replace(r'\end{thebibliography}',BEGIN+'\n'+bibliography+END+'\n'+r'\end{thebibliography}',1)
    source=source.replace(r'\begin{thebibliography}{9}',r'\begin{thebibliography}{99}')
    sources=json.loads((DATA/'paper_evaluation_sources.json').read_text())
    sources.update(json.loads((DATA/'common_benchmark_sources.json').read_text()))
    source_plan=json.loads(original.PLANFILE.read_text())
    appendix=BEGIN+'\n'+r'\newpage\section{Local-FDR Comparisons on the Same Test Datasets}'+'\n'
    kind=None
    for page,group in enumerate(groups[1:],2):
        if kind!=group['kind']:
            kind=group['kind']
            appendix+=r'\subsection{'+('Simulated test datasets' if kind=='synthetic' else 'Real predictors with simulated responses')+'}\n'
        old=old_groups[group['original_group_index']]
        appendix+='\n'+r'\begin{figure}[H]'+'\n'+original.description(old,source_plan,sources)+'\n'
        appendix+=r'\par\medskip\centering'+'\n'+r'\includegraphics[page='+str(page)+r',width=\linewidth]{lfdr_comparison_figures.pdf}'+'\n'
        appendix+=r'\caption{Paired local-FDR comparison on the same fixed assessment draws.}'+'\n'
        appendix+=r'\label{fig:lfdr-'+str(page-1)+'}\n'+r'\end{figure}'+'\n'
    appendix+=END+'\n'
    assert source.count(r'\end{document}')==1
    return source.replace(r'\end{document}',appendix+r'\end{document}')


def render():
    assessment.check_frozen()
    audit=json.loads((DATA/'lfdr_comparison_audit.json').read_text())
    assert audit['passed'] and audit['datasets']==4000 and audit['exact_rerun_cases']==list(range(40))
    assert audit['counts_sha256']==sha(assessment.COUNTS) and audit['results_sha256']==sha(assessment.RESULTS)
    result=json.loads(assessment.RESULTS.read_text());groups,old_groups=groups_from_counts()
    write_json(PLOTDATA,groups);draw(groups)
    (ROOT/'final_model_note.tex').write_text(build_source(groups,old_groups,result))
    checks=compile_note();assert len(checks['actual_pdf_figure_checks'])==49
    with (DATA/'lfdr_comparison_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['scope','case','method','target','tpr','fdr','tpr_se','fdr_se'])
        records=[('overall',-1,result['overall'])]+[('setting',i,s['methods']) for i,s in enumerate(result['settings'])]
        for scope,case,methods in records:
            for method,d in methods.items():
                for i,target in enumerate(result['targets']):
                    writer.writerow([scope,case,method,target]+[d[k][i] for k in ['tpr','fdr','tpr_se','fdr_se']])
    verification=json.loads((DATA/'current_note_verification.json').read_text())
    original_methods=['the_model','gb_ipss','lasso_ipss','lasso_cv']
    verification.update(note_source_sha256=sha(ROOT/'final_model_note.tex'),pdf_sha256=sha(ROOT/'final_model_note.pdf'),
        renderer_sha256=sha(Path(__file__).with_name('render_current_note.py')),
        original_displayed_methods=original_methods,lfdr_displayed_methods=assessment.METHODS,
        displayed_methods=original_methods+assessment.METHODS[1:],
        paired_figures=49,main_figures=9,appendix_simulated_figures=16,appendix_real_predictor_figures=24,
        lfdr_comparison_datasets=4000,lfdr_comparison_figures=25,lfdr_comparison_audit_sha256=sha(DATA/'lfdr_comparison_audit.json'),
        lfdr_comparison_renderer_sha256=sha(Path(__file__)),lfdr_comparison_figure_sha256=sha(FIGURES),
        lfdr_plot_data_sha256=sha(PLOTDATA),original_curves_unchanged=True,**checks)
    write_json(DATA/'current_note_verification.json',verification)
    relocation=json.loads((ROOT/'relocation.json').read_text())
    changed=['final_model_note.tex','final_model_note.pdf','data/current_note_verification.json',
             'simulation/render_current_note.py']
    for record in relocation['files']:
        p=ROOT/record['path']
        if record['path'] in changed:
            record['packaged_sha256']=sha(p);record['packaged_bytes']=p.stat().st_size
    for mapping in relocation['frozen_files'].values():
        for relative in mapping['packaged_files']:
            if relative in changed:mapping['packaged_files'][relative]=sha(ROOT/relative)
    write_json(ROOT/'relocation.json',relocation)
    print('Appended all 24 setting comparisons and the overall figure:',checks['pages'],'pages, 49 paired figures.',flush=True)


if __name__=='__main__':render()
