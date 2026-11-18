"""Rebuild the current note from its retained source and verified plot counts."""
import os,json,re,shutil,subprocess,sys,tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from current_data import ROOT,DATA,sha,write_json
FIGURES='final_model_note_figures.pdf'
LABELS={'the_model': 'The model', 'gb_ipss': 'GB-IPSS', 'lasso_ipss': 'Lasso-IPSS', 'lasso_cv': 'LassoCV (no FDR target)'}
COLORS={'the_model': '#245b8d', 'gb_ipss': '#555555', 'lasso_ipss': '#3a8f66', 'lasso_cv': '#cf3434'}
def draw(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(ROOT/FIGURES) as pdf:
        for group in groups:
            fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
            for name,values in group['values'].items():
                for ax,metric in zip(axes,['fdr','tpr']):
                    ax.plot(group['targets'],values[metric],label=LABELS[name],color=COLORS[name],
                        linewidth=2.3 if name=='the_model' else 1.5)
            axes[0].plot([0,.5],[0,.5],'k--',linewidth=1,label='Target');axes[0].legend(fontsize=7.5,loc='best')
            for ax,label in zip(axes,['False discovery rate','True positive rate']):
                ax.set(xlabel='Target FDR',ylabel=label,xlim=(0,.5),ylim=(0,1));ax.grid(alpha=.2)
            suffix=' (1 setting)' if group['settings']==1 else f" ({group['settings']} equally weighted settings)"
            fig.suptitle(group['title']+suffix);fig.tight_layout(rect=(0,0,1,.94));pdf.savefig(fig);plt.close(fig)
def pdf_pages(path):
    data=subprocess.check_output(['pdftotext','-bbox-layout',str(path),'-']).decode()
    data=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]','',data)
    root=ET.fromstring(data);ns={'x':'http://www.w3.org/1999/xhtml'};pages=[]
    for page in root.findall('.//x:page',ns):
        lines=[]
        for line in page.findall('.//x:line',ns):
            text=' '.join(''.join(w.itertext()) for w in line.findall('x:word',ns))
            lines.append((text,{k:float(line.attrib[k]) for k in ('xMin','xMax','yMin','yMax')}))
        pages.append(lines)
    return pages

def compile_note():
    with tempfile.TemporaryDirectory(prefix='ipss-single-f-note-',dir='/tmp') as folder:
        folder=Path(folder);shutil.copyfile(ROOT/'final_model_note.tex',folder/'final_model_note.tex')
        for _ in range(2):
            result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','final_model_note.tex'],
                cwd=folder,env=dict(os.environ,TEXINPUTS=str(ROOT)+':'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if result.returncode:raise RuntimeError(result.stdout[-6000:])
        log=(folder/'final_model_note.log').read_text()
        warnings=[s for s in log.splitlines() if 'Warning:' in s or 'Overfull' in s]
        if warnings:raise RuntimeError('\n'.join(warnings))
        pages=pdf_pages(folder/'final_model_note.pdf');checks=[]
        for page_no,lines in enumerate(pages,1):
            captions=[(int(m.group(1)),box) for text,box in lines if (m:=re.match(r'^Figure (\d+):',text))]
            lower=0
            for number,caption in sorted(captions,key=lambda v:v[1]['yMin']):
                block=[(t,b) for t,b in lines if lower<b['yMin']<caption['yMin']]
                targets=[b for t,b in block if t=='Target FDR'];left=[b for t,b in block if t=='False discovery rate'];right=[b for t,b in block if t=='True positive rate']
                assert len(targets)==2 and len(left)==len(right)==1,(page_no,number)
                assert abs(targets[0]['yMin']-targets[1]['yMin'])<.01 and left[0]['xMin']<right[0]['xMin']
                above=' '.join(t for t,b in block if b['yMax']<left[0]['yMin'])
                assert 'datasets' in above,(page_no,number,above)
                checks.append(dict(figure=number,page=page_no,paired_axes=True,dataset_description_above=True));lower=caption['yMax']
        expected = (ROOT/'final_model_note.tex').read_text().count(r'\begin{figure}[H]')
        assert sorted(c['figure'] for c in checks)==list(range(1,expected+1))
        staged=ROOT/'.final_model_note.pdf.tmp';shutil.copyfile(folder/'final_model_note.pdf',staged);staged.replace(ROOT/'final_model_note.pdf')
    return dict(pages=len(pages),actual_pdf_figure_checks=checks,unresolved_compilation_warnings=warnings)

def main():
    if (DATA/'lfdr_comparison_audit.json').exists():
        from render_lfdr_comparison import render
    else:
        from render_paper_evaluation import render
    render()

if __name__=='__main__':
    main()
