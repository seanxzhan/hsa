import os
import importlib

rep = 'occflexi'
expt = 25
mode = 'asb_scaling_inv_mesh'

module = importlib.import_module(f"run.{rep}.{rep}_{expt}")
model_idx_to_anno_id = getattr(module, "model_idx_to_anno_id")
# shape_indices = [model_idx_to_anno_id[i] for i in range(start, end)]

# Path pattern for the images
base_path = f"/projects/hsa/results/{rep}/{rep}_{expt}/"
image_pattern = mode + "/{anno_str}/{parts_str}/assembly_results.png"
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
        <title>Shape Assembly (Scaled) Results</title>
        <style>
            .anno-header {
                width: 100%;
                text-align: center;
                font-size: 1.2em;
                font-weight: bold;
                padding: 10px;
                border: 2px solid black;
                margin-top: 20px;
                margin-bottom: 10px;
                box-sizing: border-box;
                cursor: pointer;
            }
            .container {
                display: none; /* Initially hide all containers */
                flex-wrap: wrap;
                justify-content: space-around;
                align-items: center;
            }
            .item {
                width: 45%;
                margin: 10px;
                text-align: center;
                border: 1px solid black;
                padding: 10px;
                box-sizing: border-box;
            }
            img {
                height: 100px;
                width: auto;
            }
            h3 {
                margin: 0;
                padding-bottom: 5px;
            }
        </style>
        <script>
            function toggleVisibility(id) {
                var container = document.getElementById(id);
                if (container.style.display === "none" || container.style.display === "") {
                    container.style.display = "flex";
                } else {
                    container.style.display = "none";
                }
            }
        </script>
    </head>
    <body>
        <h2>Shape Assembly (Scaled) Results</h2>
    """

    for idx, anno_str in enumerate(anno_strs):
        html_content += f"""
        <div class="anno-header" onclick="toggleVisibility('container-{idx}')">
            {anno_str}
        </div>
        <div class="container" id="container-{idx}">
        """
        parts_strs = os.listdir(os.path.join(base_path, mode, anno_str))
        for parts_str in parts_strs:
            image_path = image_pattern.format(anno_str=anno_str, parts_str=parts_str)
            if not os.path.exists(os.path.join(base_path, image_path)):
                image_path = placeholder_path
            html_content += f"""
                <div class="item">
                    <h3>{anno_str}_{parts_str}</h3>
                    <img src="{image_path}" alt="Shape {anno_str}_{parts_str} results">
                </div>
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
