import subprocess

p = 'chapters/04_prediction.tex'
head = subprocess.run(['git', 'show', 'HEAD:manuscripts/MAgent_Theory_2.0/chapters/04_prediction.tex'],
                      capture_output=True).stdout.decode('utf-8')
cur = open(p, encoding='utf-8', newline='').read()
start_h = head.index('\\propositionheading{Common recovery by likelihood descent}')
end_h = head.index('\\section{A predictive coding criterion}')
blk = head[start_h:end_h].replace('\r\n', '\n')
start_c = cur.index('\\propositionheading{\t')
end_c = cur.index('\\section{A predictive coding criterion}')
cur = cur[:start_c] + blk + cur[end_c:]
open(p, 'w', encoding='utf-8', newline='').write(cur)
print('tabs remaining', cur.count('\tau'))
