from matplotlib import pyplot as plt


def more_than_one_line(df, metric):
    fig, ax = plt.subplots(figsize=(14, 10))
    df.pivot(index='dates', columns='classes', values=metric).plot(ax=ax)

    title = 'Date based ' + str(metric)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='best', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(fontsize=12)
    fig = plt.gcf()

    return fig


def scatter_plot(df, metric_1, metric_2):
    _, axs = plt.subplots(1, sharex='col', figsize=(14, 10))
    axs.scatter(df[metric_1], df[metric_2])
    title = str(metric_1) + ' vs ' + str(metric_2)
    axs.set_title(title, fontsize=12)
    axs.set_xlabel(metric_1, fontsize=12)
    axs.set_ylabel(metric_2, fontsize=12)
    fig = plt.gcf()

    return fig


def hist_plot(df):
    metric_1 = df['metric_1']
    fig, ax = plt.subplots(figsize=(14, 10))
    counts, bins, bars = ax.hist(metric_1, bins=5)
    fig = plt.gcf()
    return counts, bins, bars, fig