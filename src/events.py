"""Temporal alarm rule. Only consecutive, usable dense windows create events."""
def make_events(records, threshold, consecutive=2):
    events=[]; run=[]
    def flush():
        if len(run)>=consecutive:
            events.append(dict(start=run[0]['start'],end=run[-1]['end'],
                               peak_score=max(r['score'] for r in run),status='pending'))
    for row in records:
        is_positive = row['usable'] and row['score']>=threshold
        if run and row['start'] > run[-1]['end']+0.01:
            flush(); run=[]
        if is_positive:
            run.append(row)
        else:
            flush(); run=[]
    flush()
    return events
