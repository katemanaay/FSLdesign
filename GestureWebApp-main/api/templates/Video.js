function showVideo(videoName) {
    var videoSource = '';

    // Assign the video source based on the clicked choice
    if (videoName === 'OO_1') {
        videoSource = 'OO_1.mp4'; // Replace 'video1.mp4' with the actual video source for 'OO'
    } else if (videoName === 'video2') {
        videoSource = 'video2.mp4'; // Replace 'video2.mp4' with the actual video source for 'Ano?'
    } else if (videoName === 'videoN') {
        videoSource = 'videoN.mp4'; // Replace 'videoN.mp4' with the actual video source for 'Pakiusap, pabagalin mo'
    }
    
    // Modify this part to display the video as needed
    // For example, you can use HTML5 <video> tag or display it in a modal
    // Here is an example using HTML5 <video> tag
    var videoElement = document.createElement('video');
    videoElement.setAttribute('controls', 'true');
    videoElement.setAttribute('autoplay', 'true');
    videoElement.style.width = '100%'; // Adjust video width as needed
    videoElement.style.height = 'auto'; // Adjust video height as needed
    videoElement.innerHTML = `<source src="${videoSource}" type="video/mp4">`;

    // Replace the existing content or append it to a specific element
    var videoContainer = document.getElementById('videoContainer'); // Change 'videoContainer' to your specific container ID
    videoContainer.innerHTML = ''; // Clear previous video
    videoContainer.appendChild(videoElement); // Append the new video
}