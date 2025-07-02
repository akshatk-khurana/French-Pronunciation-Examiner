/**
 * Handles logic for the practice session functionality including audio recording,
 * silence detection, word-by-word feedback, and playback of correct pronunciations.
 */

const beginBtn = document.getElementById('begin-btn');
const pauseBtn = document.getElementById('pause-btn');
const pronounceDiv = document.getElementById('pronounce-question');
const wordDiv = document.getElementById('words');
const individualWordDivs = wordDiv.children;

const question = document.getElementById('question').value;
const response = document.getElementById('response').value;

const responseArray = response.split(" ");
let wordCursor = 0;
let currentAudio = null;
let speaking = false;

let SILENCE_DELAY = 1000; // The duration for silence between words ()

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
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    const dataArray = new Uint8Array(analyser.fftSize);
    const silenceTimeoutRef = { current: null };

    const recorder = new MediaRecorder(stream);
    let chunks = [];

    const detectSilence = () => {
      analyser.getByteTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const val = (dataArray[i] - 128) / 128;
        sum += val * val;
      }
      const rms = Math.sqrt(sum / dataArray.length);

      if (recorder.state === "recording") {
        if (rms < 0.015) {
          if (!silenceTimeoutRef.current) {
            silenceTimeoutRef.current = setTimeout(() => {
              if (recorder.state === "recording") {
                recorder.stop();
              }
            }, SILENCE_DELAY);
          }
        } else {
          if (silenceTimeoutRef.current) {
            clearTimeout(silenceTimeoutRef.current);
            silenceTimeoutRef.current = null;
          }
        }
        requestAnimationFrame(detectSilence);
      }
    };

    recorder.ondataavailable = e => chunks.push(e.data);

    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const score = await scoreAudio(blob);

      if (score === "OK") {
        getCurrentWordDiv().classList.add("text-warning");
      } else if (score === "Good") {
        getCurrentWordDiv().classList.add("text-success");
      } else if (score === "Bad") {
        getCurrentWordDiv().classList.add("text-danger");
      }

      chunks = [];
      wordCursor++;

      if (speaking) {
        recorder.start();
        setTimeout(() => {
          requestAnimationFrame(detectSilence);
        }, 500);
      }
    };

    beginBtn.addEventListener("click", () => {
      speaking = true;
      chunks = [];
      recorder.start();
      setTimeout(() => {
        requestAnimationFrame(detectSilence);
      }, 500);
    });

    pauseBtn.addEventListener("click", () => {
      speaking = false;
      if (recorder.state === "recording") {
        recorder.stop();
      }
    });

  }).catch(err => {
    console.error("Mic error:", err);
  });
} else {
  console.log("getUserMedia not supported.");
}

/**
 * Returns the DOM element for the current word being practiced.
 */
function getCurrentWordDiv() {
  return document.getElementById(String(wordCursor));
}

/**
 * Returns the text of the current word being practiced.
 */
function getCurrentWord() {
  return document.getElementById(String(wordCursor)).innerHTML;
}

/**
 * Sends the recorded audio blob to the backend for scoring.
 */
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

/**
 * Retrieves and plays the pronunciation audio for a given phrase.
 */
function pronouncePhrase(phrase) {
  fetch(`/pronounce/${encodeURIComponent(phrase)}/`)
    .then(res => res.json())
    .then(data => {
      currentAudio = new Audio(data.audio_url);
      currentAudio.play();
    });
}

/**
 * Stops any currently playing audio to avoid overlapping sounds.
 */
function stopPreviousAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
}