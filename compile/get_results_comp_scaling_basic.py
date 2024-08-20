import os
import importlib

rep = 'occflexi'
expt = 19
mode = 'comp_scaling'
types = ['geom']
start = 0
end = 50

module = importlib.import_module(f"run.{rep}.{rep}_{expt}")
model_idx_to_anno_id = getattr(module, "model_idx_to_anno_id")
shape_indices = [model_idx_to_anno_id[i] for i in range(start, end)]

# Path pattern for the images
base_path = f"/projects/hsa/results/{rep}/{rep}_{expt}/"
image_pattern = mode + "/{shape_idx}/{type}/{fixed_indices}/{samp_idx}/{shape_idx}_results.png"
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
        <title>Shape Completion (Scaled) Results</title>
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
                display: none; /* Initially hide all containers */
                width: 100%;
                box-sizing: border-box;
            }
            .column {
                display: flex;
                flex-direction: column;
                align-items: center;
                width: calc(100% / 4);
                box-sizing: border-box;
                padding: 5px;
            }
            .item {
                margin: 10px;
                text-align: center;
                border: 1px solid black;
                padding: 10px;
                box-sizing: border-box;
                cursor: pointer;
                width: 90%;
            }
            img {
                width: 100%;
            }
            h3 {
                margin: 0;
                padding-bottom: 5px;
                text-align: center;
            }
            h4 {
                margin: 5px 0;
                cursor: pointer;
                color: blue;
                text-decoration: underline;
            }
            .images {
                display: flex;
                flex-direction: column;
                align-items: center;
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

            function toggleImages(id) {
                var images = document.getElementById(id);
                if (images.style.display === "none" || images.style.display === "") {
                    images.style.display = "flex";
                } else {
                    images.style.display = "none";
                }
            }
        </script>
    </head>
    <body>
        <h2 style="text-align: center;">Shape Completion (Scaled) Results</h2>
    """

    for idx, shape_idx in enumerate(shape_indices):
        html_content += f"""
        <div class="anno-header" onclick="toggleVisibility('container-{idx}')">
            {shape_idx}
        </div>
        <div class="container" id="container-{idx}">
        """
        for t in types:
            # html_content += f"""
            # <div class="column">
            #     <h3>{t}</h3>
            # """
            lst_fixed_indices = sorted(
                os.listdir(os.path.join(base_path, mode, shape_idx, t)), key=str)
            for fixed_idx, fixed_indices in enumerate(lst_fixed_indices):
                html_content += f"""
                <div class="column">
                <h4 onclick="toggleImages('images-{idx}-{t}-{fixed_idx}')">{fixed_indices}</h4>
                <div class="images" id="images-{idx}-{t}-{fixed_idx}">
                """
                samp_indices = sorted(
                    os.listdir(os.path.join(base_path, mode, shape_idx, t, fixed_indices)), key=int)
                for samp_idx in samp_indices:
                    image_path = image_pattern.format(
                        shape_idx=shape_idx, type=t, fixed_indices=fixed_indices, samp_idx=samp_idx)
                    if not os.path.exists(os.path.join(base_path, image_path)):
                        image_path = placeholder_path
                    html_content += f"""
                        <div class="item">
                            <img src="{image_path}" alt="Shape {shape_idx}_{t}_{samp_idx} results">
                        </div>
                    """
                html_content += """
                </div>
                </div>
                """
            # html_content += """
            # </div>
            # """
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
generate_html(shape_indices)
