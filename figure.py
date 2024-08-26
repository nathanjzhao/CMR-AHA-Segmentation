import matplotlib.pyplot as plt
import numpy as np
from utils.dataset import DataSet


def figure1():
    # Data
    categories = ['Overall Score', 'LV Dice Score', 'Inferior Dice Score', 'Anterior Dice Score']
    dwi_scores = [0.617, 0.8248491883277893, 0.4651156961917877, 0.430521696805954]
    dwi_md_scores = [0.617, 0.8510607481002808, 0.4651156961917877, 0.430521696805954]
    dwi_md_e1_scores = [0.552367091178894, 0.862199604511261, 0.4201417863368988, 0.33495938777923584]

    # Set up the bar chart
    width = 0.08  # Reduced width of each bar
    pair_gap = 0.2  # Gap between pairs of bars

    # Calculate positions for the bars
    positions = np.arange(len(categories)) * (3 * width + pair_gap)

    fig, ax = plt.subplots(figsize=(8, 6))  # Increased figure width
    rects1 = ax.bar(positions - width, dwi_scores, width, label='DWI', color='#3498db', alpha=0.8)
    rects2 = ax.bar(positions, dwi_md_scores, width, label='DWI + MD', color='#e74c3c', alpha=0.8)
    rects3 = ax.bar(positions + width, dwi_md_e1_scores, width, label='DWI + MD + E1', color='#2ecc71', alpha=0.8)

    # Customize the chart
    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of DWI, DWI+MD, and DWI+MD+E1 Scores', fontsize=16, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)

    # Set y-axis limit to slightly above the maximum score
    ax.set_ylim(0, max(max(dwi_scores), max(dwi_md_scores), max(dwi_md_e1_scores)) * 1.1)

    # Add a light grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add value labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Adjust layout and display the plot
    fig.tight_layout()
    plt.show()
    plt.savefig('figure.png')

def figure2():
    # Create a dataset instance
    dataset = DataSet(
        data_folder='/home/nathanzh/AHA_Segmentation/data/unswapped_data/train',
        use_mask=True,
        use_MD=True,
        use_E1=True,
        use_E1_xyz=True,
        no_midpoint=True,
        largest_size=200
    )

    # Get a random sample from the dataset
    sample_idx = np.random.randint(len(dataset))
    dwi, _, mask, md, e1, e1_xyz = dataset[sample_idx]

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot DWI
    axs[0].imshow(dwi.squeeze(), cmap='gray')
    # axs[0].set_title('DWI')
    axs[0].axis('off')

    # Plot MD
    axs[1].imshow(md.squeeze(), cmap='coolwarm')
    # axs[1].set_title('MD')
    axs[1].axis('off')

    # Plot E1
    e1_np = e1.squeeze().numpy()  
    e1_np = e1_np.transpose(1, 2, 0)
    axs[2].imshow(e1_np)
    # axs[2].set_title('E1')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig('figure2.png')
    plt.close()

def figure3():
    # Create a figure for the legend
    fig, ax = plt.subplots(figsize=(5, 3))

    # Create colored patches for the legend
    overlap_patch = plt.Rectangle((0, 0), 1, 1, fc=(1, 0, 1))  # Full red + full blue = magenta
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc=(0, 0, 1))  # Full blue
    red_patch = plt.Rectangle((0, 0), 1, 1, fc=(1, 0, 0))  # Full red


    # Add the legend to the plot
    ax.legend([overlap_patch, blue_patch, red_patch], 
              ['Overlap', 'False Positive', 'False Negative'],
              loc='center')

    # Remove axes
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figure3.png')
    plt.close()

if __name__ == "__main__":
    figure1()
    figure2()
    figure3()