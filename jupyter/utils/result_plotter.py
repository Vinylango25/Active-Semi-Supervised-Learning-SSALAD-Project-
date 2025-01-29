import seaborn as sns
import matplotlib.pyplot as plt


# function for plotting results

def plot_ssalad_results(results_df,dataset, model_name,log_scale=True):

    if "query_strategy" not in results_df.columns:
        raise Exception(f"Ensure we have `query_strategy` on the columns, Columns found {results_df.columns}")

    if "propagation" not in results_df.columns:
        raise Exception(f"Ensure we have `propagation` on the columns, Columns found {results_df.columns}")

    if "fraction" not in results_df.columns:
        raise Exception(f"Ensure we have `fraction` on the columns, Columns found {results_df.columns}")
    
    strategies = list(results_df['query_strategy'].unique())
    propagation = list(results_df['propagation'].unique())

    num_strategies = len(strategies)
    num_propagation = len(propagation)
    fractions = results_df["fraction"].unique().tolist()

    # set colors for each strategy
    colors = sns.color_palette('tab10', num_strategies)
    # set line styles for each propagation
    line_styles = ['-', '--']

    if log_scale and 0 in results_df['fraction'].unique():
        split_axis = True
    else:
        split_axis = False

    if log_scale and split_axis:

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(8, 5), width_ratios=[1, 8])
        fig.subplots_adjust(wspace=0.05)  # adjust space between Axes

        for i, strategy in enumerate(strategies):
        # plot and fill between mean and std for each propagation
            for j, prop in enumerate(propagation):

                zero_mean = results_df[(results_df['fraction'] == 0) &
                                       (results_df['query_strategy'] == strategy) &
                                       (results_df['propagation'] == prop)
                                       ].groupby('fraction').mean(numeric_only=True).reset_index()
                zero_std = results_df[(results_df['fraction'] == 0) &
                                       (results_df['query_strategy'] == strategy) &
                                       (results_df['propagation'] == prop)
                                       ].groupby('fraction').std(numeric_only=True).reset_index()

                ax1.errorbar(0, zero_mean['roc_auc'][0], yerr=zero_std['roc_auc'][0], marker='o')
                ax1.errorbar(0, zero_mean['roc_auc'][0], yerr=zero_std['roc_auc'][0], marker='o')

        
        # zoom-in / limit the view to different portions of the data
        ax1.set_xlim(-0.01, 0.01)  # outliers only
        # ax1.set_ylim(0.5, 1)
        ax2.set_xlim(0.01, 1)  # most of the data

        # hide spines between ax and ax2
        ax1.spines.right.set_visible(False)
        ax2.spines.left.set_visible(False)

        ax1.set_ylabel('AUC')
        ax1.set_xticks([0])


        d = 2  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
        # turn off the spines and remove ticks and labels for ax2
        ax2.yaxis.set_visible(False)

    else:

        fig, ax2 = plt.subplots(1, 1, figsize=(8, 5))

    for i, strategy in enumerate(strategies):
        # plot and fill between mean and std for each propagation
        for j, prop in enumerate(propagation):
            mean_result = results_df[(results_df['query_strategy'] == strategy) & (results_df['propagation'] == prop)].groupby('fraction').mean(numeric_only=True).reset_index()
            std_result = results_df[(results_df['query_strategy'] == strategy) & (results_df['propagation'] == prop)].groupby('fraction').std(numeric_only=True).reset_index()
            
            ax2.plot(mean_result['fraction'], mean_result['roc_auc'], label=strategy,
                    color=colors[i], linestyle=line_styles[j])
            ax2.fill_between(mean_result['fraction'], mean_result['roc_auc']-std_result['roc_auc'], mean_result['roc_auc']+std_result['roc_auc'],
                            alpha=0.2, color=colors[i])
            
    ax2.set_title(f'{model_name} on {dataset}')
    if not split_axis:
        ax2.set_ylabel('AUC')
    ax2.set_xlabel(f'% labels queried')
    if log_scale:
        ax2.set_xscale('log')
    ax2.set_xticks(fractions)
    ax2.set_xticklabels(['{:g}'.format(x*100) for x in fractions])
    ax2.set_xlim(fractions[0]-.01, max(fractions)+0.01)
    # ax2.set_ylim(0.5,1)

    # make legend showing strategy by color and propagation by line style
    color_markers = [plt.Line2D([0], [0], marker='s', color=colors[i], lw=0) for i in range(num_strategies)]
    prop_markers = [plt.Line2D([0], [0], color='k', lw=1, linestyle=line_styles[j]) for j in range(num_propagation)]

    markers = color_markers + prop_markers
    labels = strategies + ['propagation', 'no prop.']

    ax2.legend(markers, labels, bbox_to_anchor=(1, 1.0))
    


