import os


def save_fig(fig, name, loc='common/'):
    file_name = name + '.png'
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, loc, file_name)
    fig.savefig(path)