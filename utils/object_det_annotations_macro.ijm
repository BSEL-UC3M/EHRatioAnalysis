/*
 * Annotate ROI and Append to CSV
 * 
 * Description:
 * This macro allows the user to annotate centroids for regions of interest (ROIs) 
 * in medical images. Specifically, it captures the coordinates of the left and 
 * right ear centroids for each image. The user clicks on the left ear followed by 
 * the right ear for each image, and the results are saved in a single CSV file.
 * 
 * Author: @cfusterbarcelo
 * Creation Date: 30/12/2024
 * 
 * Usage:
 * - Load this macro in Fiji/ImageJ.
 * - Run the macro.
 * - Select the folder containing the images to annotate.
 * - Select or create the CSV file where the annotations will be saved.
 * - Follow the instructions provided in the popup windows to annotate images.
 * - The annotated data will be appended to the selected CSV file.
 */

macro "Annotate ROI and Append to CSV with Instructions" {
    // Prompt the user to select the folder for images
    inputDir = getDirectory("Choose Image Folder");

    // Prompt the user to select the single CSV file for saving annotations
    outputFile = File.openDialog("Select or create a single CSV file for annotations");

    // Check if the CSV file already exists
    if (!File.exists(outputFile)) {
        // If it doesn't exist, create it and add a header
        File.append("File Name,Left Ear X,Left Ear Y,Right Ear X,Right Ear Y\n", outputFile);
    }

    // Get list of images in the selected folder
    list = getFileList(inputDir);
    totalImages = list.length; // Total number of images

    // Display instructions to the user
    showMessage("Instructions",
        "Instructions for Annotation:\n\n" +
        "1. You will annotate each image by clicking twice:\n" +
        "   - First click: Left ear centroid.\n" +
        "   - Second click: Right ear centroid.\n" +
        "2. After clicking both points, press 'OK' to proceed to the next image.\n" +
        "3. Make sure to annotate all images accurately.\n\n" +
        "The annotations will be saved to the selected CSV file. Click 'OK' to begin.");

    // Loop through all images in the folder
    for (i = 0; i < totalImages; i++) {
        open(inputDir + list[i]);
        print("Opened image: " + list[i]);

        // Set the Multi-Point Tool
        setTool("multipoint");
        run("Point Tool...", "type=Circle color=Yellow size=[Extra Large] label counter=0");

        remainingImages = totalImages - i - 1; // Calculate how many images are left

        // Display message with the number of images left and instructions for annotation
        waitForUser("Annotate Image",
            "Click exactly two points on the image:\n" +
            "- First point: Left ear centroid.\n" +
            "- Second point: Right ear centroid.\n\n" +
            "Press 'OK' when done.\n\n" +
            "Image: " + (i + 1) + " of " + totalImages + "\n" +
            "Remaining images: " + remainingImages);

        // Retrieve coordinates of the clicked points
        getSelectionCoordinates(xpoints, ypoints);

        // Validate the number of points
        if (xpoints.length != 2) {
            showMessage("Error", "Exactly two points are required. Skipping this image.");
            close();
            continue; // Skip this image and go to the next
        }

        // Append the points to the existing CSV file
        File.append(list[i] + "," + xpoints[0] + "," + ypoints[0] + "," + xpoints[1] + "," + ypoints[1] + "\n", outputFile);

        // Close the current image
        close();
    }

    // Final confirmation message
    showMessage("Done", "Annotations from folder saved to: " + outputFile);
}
