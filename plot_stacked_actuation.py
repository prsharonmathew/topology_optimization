"""stack iteration images in a folder into a grid with title and save
        as one image.
"""
import os
from PIL import Image, ImageDraw, ImageFont

def stack_iteration_images(folder_path, output_name="iteration_****_stacked.png"):
    """
    Stack PNG images in a folder into a grid with a title and save as one image.
    Args:
        folder_path (path): path to folder containing PNG images
        output_name (str, optional): name of output image file.
                                    Defaults to "iteration_****_stacked.png".
    """
    # 1. Gather and sort files numerically
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    # This sorts by the number in the filename so 246 comes before 1326
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if x.split('.')[0].isdigit() else x)
    
    if not files:
        print("No PNG images found in the directory.")
        return

    images = [Image.open(os.path.join(folder_path, f)) for f in files]
    
    # 2. Define Layout (Grid: 2 columns)
    cols = 2
    rows = (len(images) + cols - 1) // cols
    img_w, img_h = images[0].size
    
    title_height = 120
    padding = 20
    canvas_w = (img_w * cols) + (padding * (cols + 1))
    canvas_h = (img_h * rows) + (padding * (rows + 1)) + title_height

    # 3. Create Canvas
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 4. Add Title
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except:
        font = ImageFont.load_default()
        
    title_text = "iteration_1167"
    # Center the title
    draw.text((canvas_w // 2 - 150, 20), title_text, fill=(0, 0, 0), font=font)

    # 5. Paste Images into the Grid
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        x = padding + c * (img_w + padding)
        y = title_height + padding + r * (img_h + padding)
        canvas.paste(img, (x, y))

    # 6. Save Result
    canvas.save(output_name)
    print(f"Successfully created: {output_name}")

stack_iteration_images("./4dtopopt/stacked_fig")