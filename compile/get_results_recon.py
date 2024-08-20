import os
import importlib

rep = 'occflexi'
expt = 19
mode = 'recon'; start = 0; end = 100

module = importlib.import_module(f"run.{rep}.{rep}_{expt}")
model_idx_to_anno_id = getattr(module, "model_idx_to_anno_id")
shape_indices = [model_idx_to_anno_id[i] for i in range(start, end)]

# Path pattern for the images
base_path = f"/projects/hsa/results/{rep}/{rep}_{expt}/"
image_pattern = mode + "/{shape_idx}/{shape_idx}_results.png"
disentang_pattern0 = mode + "/{shape_idx}/{shape_idx}_results_mask_1-2-3.png"
disentang_pattern1 = mode + "/{shape_idx}/{shape_idx}_results_mask_0-2-3.png"
disentang_pattern2 = mode + "/{shape_idx}/{shape_idx}_results_mask_0-1-3.png"
disentang_pattern3 = mode + "/{shape_idx}/{shape_idx}_results_mask_0-1-2.png"
placeholder_path = "none.png"

# Function to generate HTML content
def generate_html(image_indices,
                  output_file=os.path.join(
                      base_path,
                      f"results_{mode}_{start}_{end}.html")):
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
        disentang_pattern0_path = disentang_pattern0.format(shape_idx=idx)
        disentang_pattern1_path = disentang_pattern1.format(shape_idx=idx)
        disentang_pattern2_path = disentang_pattern2.format(shape_idx=idx)
        disentang_pattern3_path = disentang_pattern3.format(shape_idx=idx)
        if not os.path.exists(os.path.join(base_path, image_path)): image_path = placeholder_path
        if not os.path.exists(os.path.join(base_path, disentang_pattern0_path)): disentang_pattern0_path = placeholder_path
        if not os.path.exists(os.path.join(base_path, disentang_pattern1_path)): disentang_pattern1_path = placeholder_path
        if not os.path.exists(os.path.join(base_path, disentang_pattern2_path)): disentang_pattern2_path = placeholder_path
        if not os.path.exists(os.path.join(base_path, disentang_pattern3_path)): disentang_pattern3_path = placeholder_path
        html_content += f"""
            <tr>
                <td class="fixed-width">{idx}</td>
                <td>
                <img src="{image_path}" alt="Shape {idx} results" width="200">
                <br>
                <img src="{disentang_pattern0_path}" alt="{idx} disentang 0" width="200">
                <img src="{disentang_pattern1_path}" alt="{idx} disentang 1" width="200">
                <img src="{disentang_pattern2_path}" alt="{idx} disentang 2" width="200">
                <img src="{disentang_pattern3_path}" alt="{idx} disentang 3" width="200">
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