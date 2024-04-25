import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "AceNodes.AudioPlay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ACE_AudioPlay") {
            console.warn("ACE_AudioPlay");
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function () {
                onExecuted?.apply(this, arguments);

                // Check for "on empty queue" condition, if applicable
                if (this.widgets[0].value === "on empty queue") {
                    if (app.ui.lastQueueSize !== 0) {
                        await new Promise((r) => setTimeout(r, 500));
                    }
                    if (app.ui.lastQueueSize !== 0) {
                        return;
                    }
                }
                
                // Assuming that 'arguments[0].audio' is the waveform and 'arguments[0].sample_rate' is the sample rate
                let waveform = arguments[0].audio; // An array of floats (-1 to 1)
                let sampleRate = arguments[0].sample_rate; // The sample rate of the audio
                // console.log(waveform, sampleRate); 
                // Create AudioContext
                let audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: sampleRate});

                // Get number of channels (assumes waveform is an array of arrays if stereo)
                let numChannels = Array.isArray(waveform[0][0]) ? waveform[0][0].length : 1;
                
                // Create AudioBuffer
                let buffer = audioCtx.createBuffer(numChannels, waveform[0].length, sampleRate);
                
                // Fill the AudioBuffer
                if (numChannels == 1) {
                    buffer.getChannelData(0).set(waveform[0]);
                } else {
                    for (let i = 0; i < numChannels; i++) {
                        let channelData = waveform[0].map(ch => ch[i]);
                        buffer.getChannelData(i).set(channelData);
                    }
                }
                
                // Create a source and connect it to the buffer
                let source = audioCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(audioCtx.destination);
                
                // Set volume, if applicable. Assuming the volume is the second widget's value.
                let volume = this.widgets[1].value; 
                if (volume !== undefined) {
                    let gainNode = audioCtx.createGain();
                    gainNode.gain.value = volume;
                    source.connect(gainNode);
                    gainNode.connect(audioCtx.destination);
                }
                
                // Play the sound
                source.start();
            };
        }
    },
});
