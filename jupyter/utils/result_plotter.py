import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# function for plotting results
def plot_ssalad_results(results_df, results_path="results",log_scale=True):

    
    if "query_strategy" not in results_df.columns:
        raise Exception(f"Ensure we have `query_strategy` on the columns, Columns found {results_df.columns}")

    if "propagation" not in results_df.columns:
        raise Exception(f"Ensure we have `propagation` on the columns, Columns found {results_df.columns}")

    if "fraction" not in results_df.columns:
        raise Exception(f"Ensure we have `fraction` on the columns, Columns found {results_df.columns}")
    
    strategies = list(results_df['query_strategy'].unique())
    # sort by alphabetical order
    strategies = sorted(strategies)

    propagation = list(results_df['propagation'].unique())
    # sort by reverse alphabetical order
    propagation = sorted(propagation, reverse=True)

    fractions = results_df["fraction"].unique().tolist()
    
    dataset_names = list(results_df['dataset'].unique())
    num_strategies = len(strategies)
    num_propagation = len(propagation)

    # set colors for each strategy
    colors = sns.color_palette('tab10', num_strategies)
    # set line styles for each propagation
    line_styles = ['-', '--']

    for dataset_name in dataset_names:
        dataset_results = results_df[results_df['dataset'] == dataset_name]
        if log_scale and 0 in dataset_results['fraction'].unique():
            split_axis = True
        else:
            split_axis = False

        if log_scale and split_axis:

            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(8, 3), width_ratios=[1, 8])
            fig.subplots_adjust(wspace=0.05)  # adjust space between Axes

            for i, strategy in enumerate(strategies):
            # plot and fill between mean and std for each propagation
                for j, prop in enumerate(propagation):

                    zero_mean = dataset_results[(dataset_results['fraction'] == 0) &
                                        (dataset_results['query_strategy'] == strategy) &
                                        (dataset_results['propagation'] == prop)
                                        ].groupby('fraction').mean(numeric_only=True).reset_index()
                    zero_std = dataset_results[(dataset_results['fraction'] == 0) &
                                        (dataset_results['query_strategy'] == strategy) &
                                        (dataset_results['propagation'] == prop)
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

            fig, ax2 = plt.subplots(1, 1, figsize=(8, 3))

        for i, strategy in enumerate(strategies):
            # plot and fill between mean and std for each propagation
            for j, prop in enumerate(propagation):
                mean_result = dataset_results[(dataset_results['query_strategy'] == strategy) & (dataset_results['propagation'] == prop)].groupby('fraction').mean(numeric_only=True).reset_index()
                std_result = dataset_results[(dataset_results['query_strategy'] == strategy) & (dataset_results['propagation'] == prop)].groupby('fraction').std(numeric_only=True).reset_index()
                
                ax2.plot(mean_result['fraction'], mean_result['roc_auc'], label=strategy,
                        color=colors[i], linestyle=line_styles[j])
                ax2.fill_between(mean_result['fraction'], mean_result['roc_auc']-std_result['roc_auc'], mean_result['roc_auc']+std_result['roc_auc'],
                                alpha=0.2, color=colors[i])
                
        num_samps = dataset_results['num_samples'].unique()[0]
        mod_name = dataset_results['model'].unique()[0]
        current_kernel = dataset_results['kernel'].unique()[0]
        ax2.set_title(f'{mod_name} on {dataset_name}, {num_samps} samples')
        if not split_axis:
            ax2.set_ylabel('AUC')
        ax2.set_xlabel(f'% labels queried')
        if log_scale:
            ax2.set_xscale('log')
        ax2.set_xticks(fractions)
        ax2.set_xticklabels(['{:g}'.format(x*100) for x in fractions])
        ax2.set_xlim(fractions[1], 1.0)
        # set y axis limits to maximum of current value or 0.5
        ymin = np.maximum(ax2.get_ylim(),0.45)[0]
        ymax = np.minimum(ax2.get_ylim(),1.05)[1]
        ax2.set_ylim(ymin,ymax)
        if split_axis:
            ax1.set_ylim(ymin,ymax)

        # make legend showing strategy by color and propagation by line style
        color_markers = [plt.Line2D([0], [0], marker='s', color=colors[i], lw=0) for i in range(num_strategies)]
        prop_markers = [plt.Line2D([0], [0], color='k', lw=1, linestyle=line_styles[j]) for j in range(num_propagation)]

        prop_labels = {True: 'propagation', False: 'no prop.'}

        markers = color_markers + prop_markers
        labels = strategies + [prop_labels[p] for p in propagation]

        ax2.legend(markers, labels, bbox_to_anchor=(1, 1.0))
        plt.tight_layout()
        saving_path = f"{results_path}/{current_kernel}/png/plt_{mod_name}_{dataset_name}.png"
        plt.savefig(f"{saving_path}")
        print(f"Plot for Model: {mod_name} and data={dataset_name}  saved at  {saving_path}")
