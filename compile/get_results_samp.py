import os
import importlib

rep = 'occflexi'
expt = 15
mode = 'samp'
types = ['occ', 'bbox', 'both']

module = importlib.import_module(f"run.{rep}.{rep}_{expt}")
model_idx_to_anno_id = getattr(module, "model_idx_to_anno_id")
# shape_indices = [model_idx_to_anno_id[i] for i in range(start, end)]

# Path pattern for the images
base_path = f"/projects/hsa/results/{rep}/{rep}_{expt}/"
image_pattern = mode + "/{shape_idx}/{type}/{samp_idx}/{shape_idx}_results.png"
placeholder_path = "none.png"

# Function to generate HTML content
def generate_html(shape_indices,
                  output_file=os.path.join(
                      base_path,
                      f"results_{mode}.html")):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shape Sampling Results</title>
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
                width: 100%;
                box-sizing: border-box;
            }
            .column {
                display: flex;
                flex-direction: column;
                align-items: center;
                width: calc(100% / 3);
                box-sizing: border-box;
                padding: 5px;
            }
            .item {
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
        <h2>Shape Sampling Results</h2>
    """

    for idx, shape_idx in enumerate(shape_indices):
        html_content += f"""
        <div class="anno-header" onclick="toggleVisibility('container-{idx}')">
            {shape_idx}
        </div>
        <div class="container" id="container-{idx}">
        """
        for t in types:
            html_content += f"""
            <div class="column">
                <h3>{t}</h3>
            """
            samp_indices = sorted(os.listdir(os.path.join(base_path, mode, shape_idx, t)), key=int)
            for samp_idx in samp_indices:
                image_path = image_pattern.format(
                    shape_idx=shape_idx, type=t, samp_idx=samp_idx)
                if not os.path.exists(os.path.join(base_path, image_path)):
                    image_path = placeholder_path
                html_content += f"""
                    <div class="item">
                        <h4>{shape_idx}_{t}_{samp_idx}</h4>
                        <img src="{image_path}" alt="Shape {shape_idx}_{t}_{samp_idx} results">
                    </div>
                """
            html_content += """
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
shape_indices = os.listdir(os.path.join(base_path, mode))
generate_html(shape_indices)
