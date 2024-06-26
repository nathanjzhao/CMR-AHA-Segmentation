import matplotlib.pyplot as plt
import numpy as np


def display_mask(mask, image, name="true"):

    mask_to_display = mask.cpu().numpy() 

    # Display the selected mask
    plt.imshow(mask_to_display, cmap='gray')  # Use grayscale color map for better visualization
    plt.title(f'mask_{name}')
    plt.axis('off')  # Hide axis for better visualization
    plt.show()
    plt.savefig(f'mask_{name}.png', bbox_inches='tight', pad_inches=0)


    image_to_display = image.cpu().numpy()  
    image_to_display = image_to_display.squeeze(0) # remove channel axis for greyscale image
    
    plt.imshow(image_to_display)  # Assuming the data is in a suitable range for display
    plt.title(f'image_{name}')
    plt.axis('off')  # Hide axis for better visualization
    plt.show()  # Display the image on the screen
    plt.savefig(f'images_{name}.png', bbox_inches='tight', pad_inches=0)  # Save the figure to a file
    plt.close()  # Close the figure to free memory