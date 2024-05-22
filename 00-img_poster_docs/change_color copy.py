from PIL import Image


from PIL import Image

# Function to change the color of the robot to the specified color
def change_robot_color(image_path, new_color):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    datas = img.getdata()

    new_image_data = []
    for item in datas:
        # Identify pixels that are black or in shades of black
        if item[0] in range(0, 100) and item[1] in range(0, 100) and item[2] in range(0, 100):
            # Change to the specified new color, preserving the alpha transparency
            new_image_data.append((*new_color, item[3]))
        else:
            new_image_data.append(item)

    img.putdata(new_image_data)
    return img

# Define the path to the original robot image
original_img_path = '/Users/lucia/Desktop/phone.png'


# Define the colors to change to
colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'white': (255, 255, 255),
    'yellow': (255, 255, 0),
    'blue': (0, 0, 255)
}

# Dictionary to store the paths to the new images
new_image_paths = {}

# Process the image for each color
for color_name, color_value in colors.items():
    # Change the robot color
    new_img = change_robot_color(original_img_path, color_value)
    # Save the new image
    new_img_path = f'/Users/lucia/Desktop/{color_name}_phone.png'
    new_img.save(new_img_path)
    # Store the new image path
    new_image_paths[color_name] = new_img_path
