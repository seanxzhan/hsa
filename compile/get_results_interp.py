import os
import importlib

rep = 'occflexi'
expt = 19
mode = 'interp'
# start = 0
# end = 50

module = importlib.import_module(f"run.{rep}.{rep}_{expt}")
model_idx_to_anno_id = getattr(module, "model_idx_to_anno_id")
# shape_indices = [model_idx_to_anno_id[i] for i in range(start, end)]

# Path pattern for the images
base_path = f"/projects/hsa/results/{rep}/{rep}_{expt}/"
image_pattern = mode + "/{anno_str}/{interp_idx}/{shape_idx}_results.png"
placeholder_path = "none.png"

# Function to generate HTML content
def generate_html(anno_strs,
                  output_file=os.path.join(
                      base_path,
                      f"results_{mode}.html")):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shape Interpolation Results</title>
        <style>
            .anno-header {
                width: 100%;
                text-align: center;
                font-size: 1.2em;
                font-weight: bold;
                padding: 10px;
                border: 2px solid black;
                margin: 20px auto;
                box-sizing: border-box;
                cursor: pointer;
            }
            .container {
                display: flex;
                width: 100%;
                box-sizing: border-box;
            }
            img {
                width: 9%;
            }
        </style>
    </head>
    <body>
        <h2 style="text-align: center;">Shape Interpolation Results</h2>
    """

    for idx, anno_str in enumerate(anno_strs):
        html_content += f"""
        <div class="anno-header">
            {anno_str}
        </div>
        <div class="container" id="container-{idx}">
        """
        interp_indices = sorted(
            os.listdir(os.path.join(base_path, mode, anno_str)), key=int)
        src_shape_idx = anno_str.split('-')[0]
        for interp_idx in interp_indices:
            image_path = image_pattern.format(
                anno_str=anno_str, interp_idx=interp_idx, shape_idx=src_shape_idx)
            if not os.path.exists(os.path.join(base_path, image_path)):
                image_path = placeholder_path
            html_content += f"""
                <img src="{image_path}" alt="Shape {anno_str}_{interp_idx} results">
            """
        html_content += """
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(output_file, "w") as file:
        file.write(html_content)

    print(f"HTML file '{output_file}' generated successfully.")

# Generate the HTML file
anno_strs = os.listdir(os.path.join(base_path, mode))
generate_html(anno_strs)
