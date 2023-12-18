function showVideo(videoFileName) {
    var modal = document.getElementById('myModal');
    var videoSource = document.getElementById('videoSource');

    // Define the directory where your videos are stored (adjust this path according to your file structure)
    var videoDirectory = "C:\designfsl\FSLdesign\GestureWebApp-main\api\static";

    // Set the source of the video dynamically based on the choice selected
    videoSource.src = videoDirectory + videoFileName;
    
    // Update the video player with the new source
    var videoPlayer = document.getElementById('videoPlayer');
    videoPlayer.load();

    // Show the modal
    modal.style.display = 'block';
}

// Example function calls
// These function calls should correspond to your choices with respective video file names
function showVideoChoice1() {
    showVideo('OO_1.mp4'); // Change 'video1.mp4' to the actual file name for choice 1
}

function showVideoChoice2() {
    showVideo('Ano_1.mp4'); // Change 'video2.mp4' to the actual file name for choice 2
}

// Add more similar functions for other choices as needed...
