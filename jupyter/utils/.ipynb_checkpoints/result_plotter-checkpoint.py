import seaborn as sns
import matplotlib.pyplot as plt

# function for plotting results
def plot_ssalad_results(results_df, dataset, model_name,
                        strategies=None, propagation=None,
                        log_scale=True, ax=None):
    
    if strategies is None:
        strategies = list(results_df['query_strategy'].unique())

    if propagation is None:
        propagation = list(results_df['propagation'].unique())
    
    num_strategies = len(strategies)
    num_propagation = len(propagation)

    fractions = results_df["fraction"].unique().tolist()

    # set colors for each strategy
    colors = sns.color_palette('tab10', num_strategies)
    # set line styles for each propagation
    line_styles = ['-', '--']

    for i, strategy in enumerate(strategies):
        for j, prop in enumerate(propagation):
            mean_result = results_df[(results_df['query_strategy'] == strategy) & (results_df['propagation'] == prop)].groupby('fraction').mean(numeric_only=True).reset_index()
            std_result = results_df[(results_df['query_strategy'] == strategy) & (results_df['propagation'] == prop)].groupby('fraction').std(numeric_only=True).reset_index()

            ax.plot(mean_result['fraction'], mean_result['roc_auc'], label=strategy,
                    color=colors[i], linestyle=line_styles[j])
            ax.fill_between(mean_result['fraction'], mean_result['roc_auc']-std_result['roc_auc'], mean_result['roc_auc']+std_result['roc_auc'],
                            alpha=0.2, color=colors[i])

    ax.set_title(f'{model_name} on {dataset}')
    ax.set_ylabel('AUC')
    ax.set_xlabel(f'% labels queried')
    
    if log_scale:
        ax.set_xscale('log')
    ax.set_xticks(fractions)
    # ax.set_xticklabels([f'{x * 100:.0f}%' if x >= 0.1 or x == 0 else '' for x in fractions])
    ax.set_xticklabels(['{:g}'.format(x*100) for x in fractions])
    ax.set_xlim(-.01, max(fractions)+0.02)
    color_markers = [plt.Line2D([0], [0], marker='s', color=colors[i], lw=0) for i in range(num_strategies)]
    prop_markers = [plt.Line2D([0], [0], color='k', lw=1, linestyle=line_styles[j]) for j in range(num_propagation)]

    markers = color_markers + prop_markers
    labels = strategies + ['propagation', 'no prop.']
    # ax.legend(markers, labels, loc='best')
    ax.legend(markers, labels, bbox_to_anchor=(1, 1.0))
    
