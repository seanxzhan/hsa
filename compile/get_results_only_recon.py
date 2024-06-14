import os
import importlib

expt = 83
module = importlib.import_module(f"run.local.local_{expt}")
model_idx_to_anno_id = getattr(module, "model_idx_to_anno_id")
# from run.local.local_77 import model_idx_to_anno_id

start = 0
end = 100
shape_indices = [model_idx_to_anno_id[i] for i in range(start, end)]

# Path pattern for the images
base_path = f"/projects/hsa/results/local_{expt}-bs-10/64/"
image_pattern = "{shape_idx}/{shape_idx}_results.png"
placeholder_path = "none.png"

# Function to generate HTML content
def generate_html(image_indices,
                  output_file=os.path.join(base_path,
                                           f"shape_reconstruction_results_{start}_{end}.html")):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shape Reconstruction Results</title>
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
        <h2>Shape Reconstruction Results</h2>
        <table>
            <tr>
                <th class="fixed-width">Shape Index</th>
                <th>Image</th>
            </tr>
    """

    for idx in image_indices:
        image_path = image_pattern.format(shape_idx=idx)
        if not os.path.exists(os.path.join(base_path, image_path)): image_path = placeholder_path
        html_content += f"""
            <tr>
                <td class="fixed-width">{idx}</td>
                <td>
                <img src="{image_path}" alt="Shape {idx} results" width="200">
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
generate_html(shape_indices)