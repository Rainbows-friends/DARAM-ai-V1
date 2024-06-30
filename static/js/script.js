document.addEventListener("DOMContentLoaded", function () {
    const videoElement = document.getElementById("video");
    setInterval(function () {
        videoElement.src = "/video_feed?" + new Date().getTime();
    }, 100);
});