import { exportWAV } from "./record.js";

const TERM = 2000 // 2 sec
const COUNT = 300 // 요청 회수

// dom
const $result = document.querySelector(".result");
const $modal = document.querySelector(".modal");
const $btn = document.querySelector("button");
const $emotionImg = document.querySelector(".emotion_img");

let data = [0]; // 음성 분석 결과
const ALL = [3, 2, 3, 2, 3, 1, 2, 1, 1]; // 더미 음성분석 결과
let num = 0;
let failedToGetUserMedia = false; // 마이크 접근실패 여부
let recBuffers = [[], []]; // 음성 데이터
let recLength = 0; // 음성데이터 길이
let numChannels = 2; // 서라운드?
let constraints = {
  // 오디오 옵션
  audio: true,
};
let listening = false; // 녹음중
let myChart;

if (navigator.getUserMedia) {
  navigator.getUserMedia(
    constraints,
    (stream) => {
      init(stream);
    },
    (err) => {
      alert("마이크가 인식되지 않습니다");
      failedToGetUserMedia = true;
    }
  );
} else if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices
    .getUserMedia(constraints)
    .then((stream) => {
      init(stream);
    })
    .catch((err) => {
      alert("마이크가 인식되지 않습니다");
      failedToGetUserMedia = true;
    });
} else failedToGetUserMedia = true;

const init = (stream) => {
  let audioContext = new AudioContext();
  let source = audioContext.createMediaStreamSource(stream);
  let context = source.context;
  let node = (
    context.createScriptProcessor || context.createJavaScriptNode
  ).call(context, 4096, numChannels, numChannels);
  node.onaudioprocess = (e) => {
    if (!listening) return;

    for (var i = 0; i < numChannels; i++) {
      recBuffers[i].push([...e.inputBuffer.getChannelData(i)]);
    }

    recLength += recBuffers[0][0].length;
  };
  source.connect(node);
  node.connect(context.destination);
};

const api = async (formData) => {
  const config = {
    headers: { "content-type": "multipart/form-data" },
  };

    const data = await axios.post("http://127.0.0.1:5000/main", formData, config);
  return data;
};

const getEmotion = async () => {
  const wav = exportWAV(recBuffers, recLength, numChannels);
  recBuffers = [[], []];
  recLength = 0;
  const formData = new FormData();
  formData.append("file", wav);
  const  data  = await api(formData);
  return data;
};

const ran = () => {
  return new Promise((resolve, reject) => {
    const num = Math.round(Math.random() * (3 - 1) + 1);
    resolve(num);
  });
};

let labels = [];
for (let i = 2; i <= 600; i += 2) {
  labels.push(i);
}

// 그래프 생성
const ctx = document.getElementById("myChart").getContext("2d");
const drawGraph = () => {
  myChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      //labels: [2,4,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60],

      datasets: [
        {
          // label: "판매량",
          data: data,
          fill: false,
          borderColor: "#ED560B",
          tension: 0.1,
        },
      ],
    },
    options: {
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        x: {
          grid: {
            display: false,
          },
        },
        y: {
          beginAtZero: true,
          ticks: {
            stepSize: 1,
            max: 3,
          },
          max: 3,
        },
      },
    },
  });
  const interval = setInterval(async () => {
    const emotion = await getEmotion();
     const result  = (emotion.data.result_lavel);
    data.push(result);
    $emotionImg.src = `../img/emotion_${result}.jpg`;

    myChart.update();
    if (data.length >= COUNT) {//여기가 총 길ㅣㅏㅂ비당아
      clearInterval(interval); // 인터벌 종료
      listening = false;
    }
  }, TERM); //////////////////
};

// 3초마다 데이터 추가 (30초 동안)

// 이벤트 리스너
$btn.addEventListener("click", () => {
  if (failedToGetUserMedia) {
    alert("마이크를 인식시켜주세요");
    return;
  }
  $result.style.display = "block";
  listening = true;
  drawGraph();
});
