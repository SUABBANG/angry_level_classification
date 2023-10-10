const mergeBuffers = (buffers, len) => {
  let result = new Float32Array(len);
  let offset = 0;
  for (var i = 0; i < buffers.length; i++) {
    result.set(buffers[i], offset);
    offset += buffers[i].length;
  }
  return result;
};

const interleave = (inputL, inputR) => {
  let len = inputL.length + inputR.length;
  let result = new Float32Array(len);

  let index = 0;
  let inputIndex = 0;

  while (index < len) {
    result[index++] = inputL[inputIndex];
    result[index++] = inputR[inputIndex];
    inputIndex++;
  }

  return result;
};

const floatTo16BitPCM = (output, offset, input) => {
  for (var i = 0; i < input.length; i++, offset += 2) {
    var s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
};

const writeString = (view, offset, string) => {
  for (var i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

const encodeWAV = (samples, numChannels) => {
  var buffer = new ArrayBuffer(44 + samples.length * 2);
  var view = new DataView(buffer);
  /* RIFF identifier */
  writeString(view, 0, "RIFF");
  /* file length */
  view.setUint32(4, 36 + samples.length * 2, true);
  /* RIFF type */
  writeString(view, 8, "WAVE");
  /* format chunk identifier */
  writeString(view, 12, "fmt ");
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count */
  view.setUint16(22, numChannels, true);
  /* sample rate */
  view.setUint32(24, 48000, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, 48000 * 4, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, numChannels * 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, "data");
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);

  return view;
};
export const exportWAV = (recBuffers, recLength, numChannels) => {
  let buffers = [];
  for (var i = 0; i < numChannels; i++) {
    buffers.push(mergeBuffers(recBuffers[i], recLength));
  }

  let interleaved =
    numChannels == 2 ? interleave(buffers[0], buffers[1]) : buffers[0];
  let dataView = encodeWAV(interleaved, numChannels);
  let blob = new Blob([dataView], { type: "audio/wav" });
  blob.name = Math.floor(new Date().getTime() / 1000) + ".wav";

  return blob;
};
