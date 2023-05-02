def format_interval(t):
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
    else:
        return '{0:02d}:{1:02d}'.format(m, s)


def decode_format_dict(fm):
	elapsed_str = format_interval(fm['elapsed'])
	remaining = (fm['total'] - fm['n']) / fm['rate'] if fm['rate'] and fm['total'] else 0
	remaining_str = format_interval(remaining) if fm['rate'] else '?'

	inv_rate = 1 / fm['rate'] if fm['rate'] else None
	rate_noinv_str = ('{0:5.2f}'.format(fm['rate']) if fm['rate'] else '?') + 'it/s'
	rate_inv_str = ('{0:5.2f}'.format(inv_rate) if inv_rate else '?') + 's/it'
	rate_str = rate_inv_str if inv_rate and inv_rate > 1 else rate_noinv_str

	ratio = round(fm['n'] / fm['total'] * 100)

	return '{:0>3d}%, {}/{}<br>{}&lt{}, {}'.format(ratio, fm['n'], fm['total'], elapsed_str, remaining_str, rate_str)
