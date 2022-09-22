import matplotlib.pyplot as plt
import numpy as np

def quick_subplots(data, ):
    pass

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def group_boxplot(data_group1, data_group2,labels_list,title,legend:list):
    # --- Labels for your data:
    xlocations  = range(len(data_group1))
    width = 0.3
    symbol= 'r+'


    ax = plt.gca()

    ax.set_xticklabels( labels_list, rotation=0)
    ax.grid(True, linestyle='dotted')
    ax.set_axisbelow(True)
    ax.set_xticks(xlocations)
    plt.xlabel('latent variables')
    plt.ylabel('latent variables value')
    plt.title(f'{title}')

    # --- Offset the positions per group:
    positions_group1 = [x-(width+0.015) for x in xlocations]
    positions_group2 = xlocations

    a=plt.boxplot(data_group1.T,
                sym=symbol,
                labels=['']*len(labels_list),
                positions=positions_group1,
                widths=width,
    #           notch=False,
    #           vert=True,
    #           whis=1.5,
    #           bootstrap=None,
    #           usermedians=None,
    #           conf_intervals=None,
    #           patch_artist=False,
                )

    b=plt.boxplot(data_group2.T,
                labels=labels_list,
                sym=symbol,
                positions=positions_group2,
                widths=width,
    #           notch=False,
    #           vert=True,
    #           whis=1.5,
    #           bootstrap=None,
    #           usermedians=None,
    #           conf_intervals=None,
    #           patch_artist=False,
                )
    set_box_color(a, 'b')
    set_box_color(b, 'r')

    hB, = plt.plot([1, 1], 'b-')
    hR, = plt.plot([1, 1], 'r-')
    plt.legend((hB, hR), tuple(legend),loc='center left', bbox_to_anchor=(1, 0.5))
    hB.set_visible(False)
    hR.set_visible(False)
    plt.tight_layout()
    #plt.savefig('boxplot_grouped.png')
    #plt.savefig('boxplot_grouped.pdf')    # when publishing, use high quality PDFs
    plt.show()                   # uncomment to show the plot.


def group_barplot(data_group1, data_group2,labels_list, title,legend:list):
    width = 0.35  # the width of the bars
    x = np.arange(len(labels_list))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, data_group1, width, label=legend[0])
    rects2 = ax.bar(x + width / 2, data_group2, width, label=legend[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('average Forbenius norm')
    ax.set_xticks(x, labels_list)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(title)

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()


def group_scatter(data_group1,data_group2,labels,title):
    assert len(data_group1)==len(data_group2)==len(labels)
    data_length=len(data_group1)
    fig, ax = plt.subplots(1, data_length,figsize=(2*data_length,2))
    for i, item_pack in enumerate(zip(data_group1,data_group2)):
        data_1=item_pack[0]
        data_2=item_pack[1]
        ax[i].scatter(data_1,data_2,s=1)
        ax[i].set_xlabel(f'original {labels[i]}')
        ax[i].set_ylabel(f'estimate {labels[i]}')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def all_var_bars(x1v1,x1vs,x1v2,x2v1,x2vs,x2v2):
    '''
    all inputs should be a time series x hidden size or 1 x hidden size
    :return:
    '''
    assert x1v1.shape==x1vs.shape==x1v2.shape==x2v1.shape==x2vs.shape==x2v2.shape
    if len(x1v1.shape)==1:
        x1v1=x1v1[None,:]
        x1vs = x1vs[None, :]
        x1v2 = x1v2[None, :]
        x2v1=x2v1[None,:]
        x2vs = x2vs[None, :]
        x2v2 = x2v2[None, :]

    x1v1=x1v1[-1,:]
    x1vs = x1vs[-1, :]
    x1v2 = x1v2[-1, :]
    x2v1=x2v1[-1,:]
    x2vs = x2vs[-1, :]
    x2v2 = x2v2[-1, :]
    hidden_size=len(x1v1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

    x_pos = np.arange(hidden_size)
    fig.suptitle("varaince of individual variance at last epoch")
    ax1.bar(x_pos, x1v1, color='r')
    x_pos = np.arange(hidden_size) + hidden_size + 1
    ax1.bar(x_pos, x1vs, color='b')
    x_pos = np.arange(hidden_size) + 2 * hidden_size + 2
    ax1.bar(x_pos, x1v2, color='g')
    ax1.set_ylabel('variance')
    ax1.legend(['x1v1', 'x1vs', 'x1v2'])
    ax1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    # ax1.tittle('individual var(X1) at last epoch')
    x_pos = np.arange(hidden_size)
    ax2.bar(x_pos, x2v1, color='r')
    x_pos = np.arange(hidden_size) + hidden_size + 1
    ax2.bar(x_pos, x2vs, color='b')
    x_pos = np.arange(hidden_size) + 2 * hidden_size + 2
    ax2.bar(x_pos, x2v2, color='g')
    ax2.legend(['x2v1', 'x2vs', 'x2v2'])
    ax2.set_ylabel('variance')
    ax2.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    plt.show()