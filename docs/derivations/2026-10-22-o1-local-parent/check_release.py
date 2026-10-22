"""Check structural release consistency and the read-only source snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
VAULT = Path('C:/Users/chris and christine/Desktop/Research')
VALIDATOR = Path('C:/Users/chris and christine/.agents/skills/rigorous-theory-search/scripts/validate_run.py')


def main() -> None:
    manifest = json.loads((ROOT/'source-manifest.json').read_text(encoding='utf-8'))
    binding = {k:manifest[k] for k in ('schema_version','contract_id','target_digest')}
    refresh_path = ROOT/'source-refresh.json'
    refresh = json.loads(refresh_path.read_text(encoding='utf-8')) if refresh_path.exists() else {'reviewed_changes':[]}
    reviewed = {r['path']:r for r in refresh['reviewed_changes']}
    checked = []
    for source in manifest['sources']:
        path = Path(source['path'])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        revision = reviewed.get(str(path),{})
        accepted = actual == source['sha256'] or (revision.get('baseline_sha256') == source['sha256'] and revision.get('current_sha256') == actual and bool(revision.get('reason')))
        checked.append({'path':str(path),'unchanged':actual == source['sha256'],'source_revision_reviewed':accepted,'sha256':actual})
    result = subprocess.run([sys.executable,str(VALIDATOR),str(ROOT),'--mode','release'],capture_output=True,text=True)
    current_vault = subprocess.check_output(['git','status','--porcelain'],cwd=VAULT,text=True)
    report = {**binding,'validator':str(VALIDATOR),'validator_exit_code':result.returncode,'validator_stdout':result.stdout,'validator_stderr':result.stderr,'structural_validation_passed':result.returncode==0,'source_checks':checked,'all_read_sources_unchanged':all(bool(item['unchanged']) for item in checked),'all_source_revisions_reviewed':all(bool(item['source_revision_reviewed']) for item in checked),'vault_status_unchanged':current_vault==manifest['vault_status_before'],'vault_head_after':subprocess.check_output(['git','rev-parse','HEAD'],cwd=VAULT,text=True).strip(),'vault_status_after':current_vault,'repo_status_after':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),'limitations':'Structural consistency and source revision review only. No mechanical theorem proof or independent verification.'}
    (ROOT/'validation-receipt.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('structural_validation_passed','validator_exit_code','all_read_sources_unchanged','all_source_revisions_reviewed','vault_status_unchanged','limitations')},indent=2))
    if result.returncode:
        print(result.stdout+result.stderr)
        raise SystemExit(result.returncode)
    if not report['all_source_revisions_reviewed']:
        raise SystemExit('Source revision drift; review affected claims.')


if __name__ == '__main__':
    main()
