import cv2
import numpy as np

def create_grids_with_overlap(image):
    # Read the image
    # image = cv2.imread(image_path)
    # if image is None:
    #     print("Error: Unable to read image.")
    #     return

    # Get the image dimensions
    height, width, _ = image.shape


    #
    # # Calculate the grid dimensions
    # if (height > width):
    #     if(height<1200):
    #         num_cols = 2
    #         overlap_ratio=0.2
    #
    #     elif(height<1700):
    #         num_cols = 3
    #         overlap_ratio=0.15
    #
    #     elif(height<2800):
    #         num_cols = 3
    #         overlap_ratio=0.1
    #
    #     else:
    #         num_cols = 4
    #         overlap_ratio=0.00
    #     num_rows = round(height / width) * num_cols
    #     if(num_rows/num_cols)>4:
    #         num_rows=num_cols*4
    #
    # else:
    #
    #     if(width<1200):
    #         num_rows = 2
    #         overlap_ratio=0.2
    #
    #     elif(width<1700):
    #         num_rows = 3
    #         overlap_ratio=0.15
    #
    #     elif(width<2800):
    #         num_rows = 3
    #         overlap_ratio=0.1
    #     else:
    #         num_rows = 4
    #         overlap_ratio=0.0
    #
    #     num_cols = round(width / height) * num_rows
    #     if(num_cols/num_rows)>4:
    #         num_cols=num_rows*4

    import math
    MaxInferecesize=640
    num_rows=math.ceil(height/MaxInferecesize)
    num_cols = math.ceil(width / MaxInferecesize)

    if(num_rows<1):
        num_rows=1
    if(num_cols<1):
        num_cols=1

    if(num_rows<=2 and num_cols<=2):
        overlap_ratio=0.2
    elif (num_rows <= 3 and num_cols <= 3):
        overlap_ratio = 0.15
    elif (num_rows <= 4 and num_cols <= 4):
        overlap_ratio = 0.1
    else:
        overlap_ratio = 0.05








    print(num_rows , "   :   ",num_cols,"    :   ",image.shape)

    # Calculate the grid cell size with the overlapping pixels
    cell_height = int(height / num_rows)
    cell_width = int(width / num_cols)
    overlap_height = int(cell_height * overlap_ratio)
    overlap_width = int(cell_width * overlap_ratio)

    # Initialize a list to store the grid cells
    grid_cells = []
    GridLocations=[]

    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the starting and ending coordinates of the grid cell
            y_start = max(0, row * cell_height - overlap_height)
            y_end = min(height, (row + 1) * cell_height + overlap_height)
            x_start = max(0, col * cell_width - overlap_width)
            x_end = min(width, (col + 1) * cell_width + overlap_width)

            # Extract the grid cell and append it to the list
            # print(y_start , "  : ",y_end , "   ", x_start,"   :  ",x_end)
            grid_cell = image[y_start:y_end, x_start:x_end]
            GridLocations.append([(y_start,y_end),(x_start,x_end)])
            grid_cells.append(grid_cell)
            print(grid_cell.shape)

    return grid_cells,GridLocations

# Example usage
# image_path = "Beachimage628.jpg"
# grid_cells = create_grids_with_overlap(image_path)
# print(len(grid_cells))
#
# # Display the grid cells
# for i, cell in enumerate(grid_cells):
#     cv2.namedWindow(f"Grid Cell {i+1}", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(f"Grid Cell {i+1}", 600, 600)
#     cv2.imshow(f"Grid Cell {i+1}", cell)
#     cv2.waitKey(0)
# #
# # cv2.destroyAllWindows()