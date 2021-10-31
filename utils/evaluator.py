import os
import codecs


def evaluate(output_dir, flag, gold_answers, pred_answers):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ref_path = os.path.join(output_dir, '%s.ref' % flag)
    pred_path = os.path.join(output_dir, '%s.pred' % flag)
    score_path = os.path.join(output_dir, '%s.score' % flag)
    temp_path = os.path.join(output_dir, '%s.temp' % flag)

    with codecs.open(ref_path, 'w', 'utf8') as ref_file:
        for ans in gold_answers:
            ref_file.write('%s\n' % '  '.join(ans))

    with codecs.open(pred_path, 'w', 'utf8') as pred_file:
        for ans in pred_answers:
            pred_file.write('%s\n' % '  '.join(ans))

    os.system('echo > %s' % temp_path)
    os.system('%s  %s %s %s > %s' % ('../utils/score.perl', temp_path, ref_path, pred_path, score_path))
    os.system('tail -n 7 %s > %s' % (score_path, temp_path))

    eval_lines = [l.rstrip() for l in codecs.open(temp_path, 'r', 'utf8')]

    # Remove temp files.
    os.remove(ref_path)
    os.remove(pred_path)
    os.remove(score_path)
    os.remove(temp_path)

    # Precision, Recall and F1 score
    return float(eval_lines[1].split(':')[1]), float(eval_lines[0].split(':')[1]), float(eval_lines[2].split(':')[1])
