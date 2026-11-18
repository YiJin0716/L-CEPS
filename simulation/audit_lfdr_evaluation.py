"""Independently audit paired inputs, selections, and reported local-FDR results."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from current_data import DATA,ROOT,sha,write_json
import lfdr_evaluation as assessment


def verify_prefix(score,selected,alpha):
    score=np.asarray(score);selected=np.asarray(selected,dtype=bool)
    if selected.any():
        cutoff=score[selected].max()
        np.testing.assert_array_equal(selected,score<=cutoff)
        assert score[selected].mean()<=alpha+1e-14
    else:cutoff=-1.
    # The first omitted score block would violate the limit. All later blocks
    # have greater scores, so no larger complete-tie prefix can be admissible.
    rest=score[~selected]
    if len(rest):
        next_set=score<=rest.min()
        assert score[next_set].mean()>alpha-1e-14,(alpha,score[next_set].mean())


def main():
    plan=assessment.check_frozen();rows=assessment.read_rows();original=assessment.reference_rows()
    assert len(rows)==len(plan['tasks'])==4000
    targets=np.asarray(plan['targets']);software=plan['software'];all_warnings={}
    errors={'lasso_gaussian_z':0.};timing=[]
    for count,(row,task) in enumerate(zip(rows,plan['tasks']),1):
        assert all(row[k]==v for k,v in task.items()),row['id']
        spec=plan['settings'][task['case']]
        engine=assessment.paper if task['origin']=='paper' else assessment.common
        x,y,truth,_=engine.generate(spec,task)
        assert hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()==row['input_sha256']
        np.testing.assert_array_equal(row['truth'],truth)
        old=original[task['origin']][task['source_id']]
        np.testing.assert_array_equal(old['truth'],truth)
        bits=np.unpackbits(row['selected_bits'],axis=-1,count=task['p']).astype(bool)
        old_bits=np.unpackbits(old['selected_bits'][0],axis=-1,count=task['p']).astype(bool)
        np.testing.assert_array_equal(bits[0],old_bits)
        np.testing.assert_array_equal(row['tp'][0],old['tp'][0]);np.testing.assert_array_equal(row['fp'][0],old['fp'][0])
        planted=np.zeros(task['p'],bool);planted[truth]=True
        np.testing.assert_array_equal(bits[:,:,planted].sum(2),row['tp'])
        np.testing.assert_array_equal(bits[:,:,~planted].sum(2),row['fp'])
        for i,alpha in enumerate(targets):
            verify_prefix(row['scores'][:,0],bits[1,i],alpha)
            verify_prefix(row['scores'][:,1],bits[2,i],alpha)
            np.testing.assert_array_equal(bits[3,i],row['scores'][:,2]<=alpha)
            verify_prefix(row['scores'][:,2],bits[4,i],alpha)
        diag=row['diagnostics']
        assert len(diag['fold'])==task['n'] and set(diag['fold'])==set(range(1,11))
        if spec['response']=='reg':
            beta=np.asarray(diag['lasso_beta']);xc=x-x.mean(0);scale=np.sqrt((xc*xc).mean(0))
            xs=xc/scale;r=y-y.mean()-xc@beta
            sigma=np.sqrt(np.sum(r*r)/(len(y)-np.count_nonzero(beta)+1))
            independent=(xs.T@r/len(y)+beta*scale)/(sigma/np.sqrt(len(y)))
            error=float(np.max(np.abs(independent-np.asarray(diag['lasso_z']))))
            errors['lasso_gaussian_z']=max(errors['lasso_gaussian_z'],error)
            np.testing.assert_allclose(independent,diag['lasso_z'],rtol=2e-5,atol=2e-5)
        timing.append(diag['seconds'])
        for message in diag['warnings']:all_warnings[message]=all_warnings.get(message,0)+1
        if count%400==0:print('Audited',count,'paired datasets',flush=True)
    with np.load(assessment.COUNTS) as a:
        assert list(a['methods'])==assessment.METHODS
        np.testing.assert_array_equal(a['tp'],np.stack([r['tp'] for r in rows],axis=1))
        np.testing.assert_array_equal(a['fp'],np.stack([r['fp'] for r in rows],axis=1))
        for case in range(40):assert (a['cases']==case).sum()==100
        report=json.loads(assessment.RESULTS.read_text())
        tpr=a['tp']/a['support'][None,:,None];fdp=a['fp']/np.maximum(1,a['tp']+a['fp'])
        for m,name in enumerate(assessment.METHODS):
            for metric,array in [('tpr',tpr),('fdr',fdp)]:
                np.testing.assert_allclose(report['overall'][name][metric],array[m].mean(0),rtol=0,atol=1e-14)
                for case,item in enumerate(report['settings']):
                    np.testing.assert_allclose(item['methods'][name][metric],array[m,a['cases']==case].mean(0),rtol=0,atol=1e-14)
    assessment.initialize();recomputed=[]
    try:
        for row in rows:
            if row['replicate']!=0:continue
            task=plan['tasks'][row['id']];new=assessment.one(task)
            np.testing.assert_array_equal(new['scores'],row['scores'])
            np.testing.assert_array_equal(new['selected_bits'],row['selected_bits'])
            recomputed.append(row['case'])
    finally:assessment.close_worker()
    assert recomputed==list(range(40))
    write_json(DATA/'lfdr_comparison_audit.json',dict(passed=True,datasets=4000,settings=40,
        exact_input_hashes=True,exact_truth=True,original_model_selections_unchanged=True,
        all_selected_sets_recounted=True,all_prefixes_valid_and_maximal=True,
        all_curves_recomputed=True,all_settings_reported=True,independent_lasso_gaussian_z_errors=errors,
        exact_rerun_cases=recomputed,software=software,warning_counts=all_warnings,
        median_comparator_seconds=float(np.median(timing)),total_comparator_seconds=float(sum(timing)),
        protocol_sha256=sha(assessment.PROTOCOL),counts_sha256=sha(assessment.COUNTS),
        results_sha256=sha(assessment.RESULTS),auditor_sha256=sha(Path(__file__))))
    print('Paired comparison audit passed, including 40 exact R reruns.',flush=True)


if __name__=='__main__':main()
