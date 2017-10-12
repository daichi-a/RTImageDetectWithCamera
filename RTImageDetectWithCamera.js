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
    let cropped_image;
    let scaled_image;
    let dataUrl;
    let ratio = 50.0 / 240.0;
    window.scaled_canvas.setAttribute("width", 50);
    window.scaled_canvas.setAttribute("height", 50);
    
    if(window.thisBrowser == 'ios' || window.thisBrowser == 'android'){
	window.scaled_ctx.drawImage(window.localVideo, 0, 80, 480, 480, 0, 0, 50, 50);
	scaled_image = window.scaled_ctx.getImageData(0, 0, 50, 50);
    }
    else{
	window.scaled_ctx.drawImage(window.localVideo, 80, 0, 480, 480, 0, 0, 50, 50);
	scaled_image = window.scaled_ctx.getImageData(0, 0, 50, 50);
    }

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
	if(window.thisBrowser == 'ios'){
	    // for iOS Devices
	    const medias = {audio: false,
			    video: {facingMode : {exact: "environment"}}};
	    navigator.getUserMedia
	    (medias,
	     function(stream){
		 window.localVideo.srcObject = stream;
	     },
	     function(error){
		 alert('navigator.getUserMedia() error:' + error);
	     });
	    
	}
	else if(window.thisBrowser == 'chrome'){
	    // for Chrome
	    let medias;
	    if(window.navigator.userAgent.toLowerCase().indexOf('android') != -1){
		// for Android Chrome (with both Front and Back Camera)
		medias = {audio: false,
			  video: {width: 480,
				  wheight: 640,
				  facingMode:{exact: "environment"}}};
	    }
	    else{
		// for PC Chrome (with only 1 Camera)
		medias = {audio: false,
			  video: {width: 640,
				  height: 480}};
		
	    }
	    navigator.mediaDevices.getUserMedia(medias)
		.then(stream => {
		    window.localVideo.src = window.URL.createObjectURL(stream);
		})
		.catch(error => {
		    console.error('navigator.mediaDvice.getUserMedia() error:', error);
		    return;
		});
	}
	else if(window.thisBrowser == 'safari'){
	    // for Mac Safari
	    const medias = {audio: false,
			    video: {width:640, 
				    height:480}};
	    navigator.getUserMedia(medias,
				   (stream) => {
				       window.localVideo.srcObject = stream;
				   },
				   (error) => {
				       console.error('navigator.getUserMedia() error:', error);
				   });
	    
	}

	setInterval(() => {
	    // Convert to ImageData, cropping, scaling for Keras Network
	    cameraToImage();
	}, 1000);
    }
}

window.addEventListener('load', () => {

    window.userAgentInLowerCase = window.navigator.userAgent.toLowerCase();
    document.getElementById("display_user_agent").innerHTML = 
	window.userAgentInLowerCase;

    if(userAgentInLowerCase.indexOf('iphone') != -1 ||
      userAgentInLowerCase.indexOf('ipad') != -1)
	window.thisBrowser = 'ios';
    if(userAgentInLowerCase.indexOf('android') != -1)
	window.thisBrowser = 'android';
    else if(userAgentInLowerCase.indexOf('chrome') != -1)
	window.thisBrowser = 'chrome';
    else if(userAgentInLowerCase.indexOf('safari') != -1)
	window.thisBrowser = 'safari';
    else
	alert('Cannot use this program for your browser');

    window.localVideo = document.getElementById('local_video');
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

