// RTImageDetectWithCamera.js
'use strict';

function prepareModel(filename_model, filename_weights, filename_metadata){
    window.model = new KerasJS.Model({
	filepaths: {
	    model: filename_model,
	    weights: filename_weights,
	    metadata: filename_metadata
	},
	gpu: true
    });
}

function predictModel(data){
    window.model.ready()
	.then(() => {
	    const inputData = {
		'input' : new Float32Array(data)
	    }
	    return model.predict(inputData)
	})
	.then(outputData => {
	    console.log(outputData['output']);
	    if(outputData['output'][0] > outputData['output'][1])
		window.resultDiv.innerHTML = '<p>りんご！</p>';
	    else
		window.resultDiv.innerHTML = '<p>みかん！</p>';
	})
	.then(err => {
	    //エラーハンドリング
	})
}

function cameraToImage(){
    window.plane_camera_canvas.setAttribute("width", 320);
    window.plane_camera_canvas.setAttribute("height", 240);
    window.plane_camera_ctx.drawImage(window.localVideo, 0, 0, 320, 240);
    let cropped_image = window.plane_camera_ctx.getImageData(40, 0, 240, 240);

    window.cropped_canvas.setAttribute("width", 240);
    window.cropped_canvas.setAttribute("height", 240);
    window.cropped_ctx.putImageData(cropped_image, 0, 0, 0, 0, 240, 240);

    window.scaled_canvas.setAttribute("width", 50);
    window.scaled_canvas.setAttribute("height", 50);
    window.scaled_ctx.scale(50.0/240.0, 50.0/240.0)
    window.scaled_ctx.drawImage(window.cropped_canvas, 0, 0);
    let scaled_image = window.scaled_ctx.getImageData(0, 0, 50, 50);

    // Uint8*4のピクセルデータをKerasに喰わせるためにfloat32に変換
    let planeCameraData = new Float32Array(scaled_image.data);
    // この時点ではRGBARGBARGBA...という感じでインターリーブで各画素が並んでいるので
    // R50*50, G50*50, B50*50という並びで学習させた
    // Kerasのmodelに喰わせるための3 * 50 * 50配列に並べ直す
    let arrayForKeras = new Float32Array(3 * 50 * 50);
    for(let x=0; x<50; x++){
	for(let y=0; y<50; y++){
	    // R
	    arrayForKeras[0 * 250 + x*50 + y] = 
		planeCameraData[x*50 + y*4 + 0];
	    // G
	    arrayForKeras[1 * 250 + x*50 + y] = 
		planeCameraData[x*50 + y*4 + 1];
	    // B
	    arrayForKeras[2 * 250 + x*50 + y] = 
		planeCameraData[x*50 + y*4 + 2];
	    // Aは破棄
	    let trash = planeCameraData[x*50 + y*4 + 3];    
	}
    }

    predictModel(arrayForKeras);

    let dataURL = window.scaled_canvas.toDataURL("image/octet-stream");    
}

function startVideo() {
    if(navigator.getUserMedia || navigator.webkitGetUserMedia || 
       navigator.mozGetUserMedia || navigator.msGetUserMedia){
	let localStream;
	
	navigator.mediaDevices.getUserMedia({video: {width:640, height:480}, audio: false})
	    .then((stream) => {
		localStream = stream;
		window.localVideo.src = window.URL.createObjectURL(localStream);
	    })
	    .catch((error) => {
		console.error('mediaDvice.getUserMedia() error:', error);
		return;
	    });
	
	setInterval(() => {
	    cameraToImage();
	}, 1000);
	
    }
    else{
	alert('getUserMedia() is not supported in your browser');
    }
}

window.addEventListener('load', () => {
    window.localVideo = document.getElementById('local_video');
    window.plane_camera_canvas = document.getElementById('plane_camera_canvas');
    window.plane_camera_ctx = window.plane_camera_canvas.getContext('2d');
    window.cropped_canvas = 
	document.getElementById('cropped_canvas')
    window.cropped_ctx = window.cropped_canvas.getContext('2d');
    window.scaled_canvas = 
	document.getElementById('scaled_canvas')
    window.scaled_ctx = window.scaled_canvas.getContext('2d');


    let filename_model = './ringo_or_mikan_model.json';
    let filename_weights = './ringo_or_mikan_weights_weights.buf';
    let filename_metadata = './ringo_or_mikan_weights_metadata.json';
    
    prepareModel(filename_model, filename_weights, filename_metadata);
    //filename_model: .json
    //filename_wights: .buf <- generated by encoder.py from hdf5
    //filename_metadata: .json <- generated by encoder.py from hdf5

    window.resultDiv = document.getElementById('predict_answer');

}, false);

window.addEventListener('unload', () => {
    
}, false);
