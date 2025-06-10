const beginBtn = document.getElementById('begin-btn');
const pauseBtn = document.getElementById('pause-btn');
const mainDiv = document.getElementById('main');
const wordDiv = document.getElementById('words');
const pronounceDiv = document.getElementById('pronounce-question');
const individualWordDivs = wordDiv.children;

const question = document.getElementById('question').value
const response = document.getElementById('response').value;

let SILENCE_DELAY = 1000;
let speaking = false;

const responseArray = response.split(" ");
let wordCursor = 0;

let currentAudio = null;

pronounceDiv.addEventListener("click", () => {
  stopPreviousAudio();
  pronouncePhrase(question);
});

Array.from(individualWordDivs).forEach(word => {
  let hoverTimeout = null;
  word.addEventListener("mouseover", () => {
    hoverTimeout = setTimeout(() => {
      stopPreviousAudio();
      pronouncePhrase(word.innerHTML);
    }, 100);
  });
  word.addEventListener("mouseout", () => {
    if (hoverTimeout) {
      clearTimeout(hoverTimeout);
      hoverTimeout = null;
    }
  });
});

if (navigator.mediaDevices.getUserMedia) {
  console.log("The mediaDevices.getUserMedia() method is supported.");

  const constraints = { audio: true };
  let chunks = [];
  let mediaRecorder = null;

  let onSuccess = function (stream) {
    mediaRecorder = new MediaRecorder(stream);

    const detectSilenceRunner = setupSilenceDetection(stream, mediaRecorder);

    beginBtn.addEventListener("click", () => {
      speaking = true;
      SILENCE_DELAY -= 900;

      mediaRecorder.start();
      requestAnimationFrame(detectSilenceRunner);
    });

    pauseBtn.addEventListener("click", () => {
      speaking = false;
      
      if (mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      }
    });

    mediaRecorder.ondataavailable = function (e) {
      chunks.push(e.data);
    };

    mediaRecorder.onstop = async function () {
      
      const blob = new Blob(chunks, { type: 'audio/webm' });
      let pronunciationScore = await scoreAudio(blob);

      if (pronunciationScore === "OK") {
        console.log("Score received")
        getCurrentWordDiv().classList.add("text-warning");
      }

      chunks = [];
      wordCursor++;

      console.log(speaking);
      if (speaking == true) {
        mediaRecorder.start();
        requestAnimationFrame(detectSilenceRunner);
      }
    };
  };

  let onError = function (err) {
    console.log("The following error occured: " + err);
  };

  navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);
} else {
  console.log("MediaDevices.getUserMedia() not supported on your browser!");
}

function getCurrentWordDiv() {
  return document.getElementById(String(wordCursor));
}

function getCurrentWord() {
  return document.getElementById(String(wordCursor)).innerHTML;
}

async function scoreAudio(blob) {
  const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
  const formData = new FormData();

  formData.append('audio', file);
  formData.append('word', getCurrentWord());

  const response = await fetch('/score/', {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();
  return data["score"];
}

function calculateRMS(dataArray) {
  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) {
    let val = (dataArray[i] - 128) / 128;
    sum += val * val;
  }
  return Math.sqrt(sum / dataArray.length);
}

function detectSilence(analyser, dataArray, mediaRecorder, silenceTimeoutRef, SILENCE_DELAY = 1500) {
  analyser.getByteTimeDomainData(dataArray);

  let rms = calculateRMS(dataArray);

  if (mediaRecorder.state === "recording") {
    if (rms < 0.01) {
      if (!silenceTimeoutRef.current) {
        silenceTimeoutRef.current = setTimeout(() => {
          if (mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            console.log("Recorder stopped due to silence.");
          }
        }, SILENCE_DELAY);
      }
    } else {
      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current);
        silenceTimeoutRef.current = null;
      }
    }
    requestAnimationFrame(() => detectSilence(analyser, dataArray, mediaRecorder, silenceTimeoutRef, SILENCE_DELAY));
  }
}

function setupSilenceDetection(stream, mediaRecorder) {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const mediaStreamSource = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  mediaStreamSource.connect(analyser);
  const dataArray = new Uint8Array(analyser.fftSize);

  const silenceTimeoutRef = { current: null };

  return () => detectSilence(analyser, dataArray, mediaRecorder, silenceTimeoutRef, SILENCE_DELAY);
}

function pronouncePhrase(phrase) {
  fetch(`/pronounce/${encodeURIComponent(phrase)}/`)
    .then(response => response.json())
    .then(data => {
      currentAudio = new Audio(data.audio_url);
      currentAudio.play();
      console.log('Received pronunciation audio for:', data.phrase);
    })
    .catch(error => {
      console.error('Error fetching pronunciation:', error);
    });
}

function stopPreviousAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
}