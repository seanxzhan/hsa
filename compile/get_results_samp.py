import os
import importlib

rep = 'occflexi'
expt = 19
mode = 'samp'; start = 0; end = 1

module = importlib.import_module(f"run.{rep}.{rep}_{expt}")
model_idx_to_anno_id = getattr(module, "model_idx_to_anno_id")
# shape_indices = ['both'] + [model_idx_to_anno_id[i] for i in range(start, end)]
shape_indices = [model_idx_to_anno_id[i] for i in range(start, end)]

# Path pattern for the images
base_path = f"/projects/hsa/results/{rep}/{rep}_{expt}/"
image_pattern = mode + "/{shape_idx}/{samp_idx}/{shape_idx}_results.png"
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
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 2px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
                width: 10px;
            }
            img {
                height: 100px;
                width: auto;
            }
            .fixed-width {
                width: 5%;
            }
        </style>
    </head>
    <body>
        <h2>Shape Sampling Results</h2>
        <table>
            <tr>
                <th class="fixed-width">Shape Index</th>
                <th>Image</th>
            </tr>
    """

    for idx, shape_idx in enumerate(shape_indices):
        html_content += f"""
            <tr>
                <td class="fixed-width">{shape_idx}</td>
                <td>
        """
        samp_indices = sorted(os.listdir(
            os.path.join(base_path, mode, shape_idx)), key=int)
        for samp_idx in samp_indices:
            image_path = image_pattern.format(
                shape_idx=shape_idx, samp_idx=samp_idx)
            if not os.path.exists(os.path.join(base_path, image_path)):
                image_path = placeholder_path
            html_content += f"""
                <img src="{image_path}" alt="Shape {shape_idx}_{samp_idx} results">
            """
        html_content += """
                </td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(output_file, "w") as file:
        file.write(html_content)

    print(f"HTML file '{output_file}' generated successfully.")

# Generate the HTML file
# shape_indices = os.listdir(os.path.join(base_path, mode))
generate_html(shape_indices)
