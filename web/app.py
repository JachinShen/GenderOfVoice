import os
from flask import Flask, request, render_template
from flask import redirect, url_for, send_from_directory
from werkzeug import secure_filename
import pickle
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import wave

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['wav', 'mp3', 'm4a', 'aac', 'ogg', 'wma'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('home.html')
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    
    if request.method=='GET':
        result=request.form
        
        cwd = os.getcwd()
        
        name, ext = os.path.splitext(filename)
        if ext != '.wav':
            # -acodec pcm_u8
            os.system("ffmpeg -y -i {0}/uploads/{1}  {0}/uploads/{2}.wav".format(cwd, filename, name))
		
        pkl_file = open('model.pkl', 'rb')
        model = pickle.load(pkl_file)
        
        scale_file = open('scale.pkl', 'rb')
        features_min_reduced, features_max_reduced = pickle.load(scale_file)
        
        wav = wave.open("{}/uploads/{}.wav".format(cwd, name) ,"rb")
        num_frame = wav.getnframes()
        num_channel = wav.getnchannels()
        framerate = wav.getframerate()
        num_sample_width = wav.getsampwidth()
        str_data = wav.readframes(num_frame)
        wav.close()
        
        wave_data = np.frombuffer(str_data, dtype = np.short)
        wave_data = wave_data.reshape(-1)
        wave_data = wave_data.T
        
        seewave = importr("seewave")
        
        wave_data_r = robjects.FloatVector(wave_data)
        songspec = seewave.spec(wave_data_r, f = framerate, plot = False)
        analysis = seewave.specprop(songspec, f = framerate, flim = robjects.FloatVector([0, 280.0/1000]), plot = False)
        ff = seewave.fund(wave_data_r, f = framerate, ovlp = 50, threshold = 5, 
                        fmax = 280, ylim=robjects.FloatVector([0, 280.0/1000]), plot = False, wl = 2048)
                        
        sd = analysis[1][0] / 1000
        Q25 = analysis[5][0] / 1000
        IQR = analysis[7][0] / 1000
        sh = analysis[12][0]
        meanfun = np.nanmean(np.array(ff).T[1])
        
        my_features_raw = np.array([IQR, meanfun, Q25, sd, sh])
        my_features = (my_features_raw - features_min_reduced) \
             / (features_max_reduced - features_min_reduced)
        
        
        prediction = model.predict_proba(my_features.reshape(1, 5))[0]
        
        if prediction[0] < 0.5:
            return render_template('result_boy.html',prediction=prediction[1] * 100)
        else:
            return render_template('result_girl.html',prediction=prediction[0] * 100)
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
                               
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
    
@app.errorhandler(400)
def page_not_found(e):
    return render_template('400.html'), 400
    
@app.errorhandler(410)
def page_not_found(e):
    return render_template('410.html'), 410
    
@app.errorhandler(500)
def page_not_found(e):
    return render_template('500.html'), 500
    
if __name__ == '__main__':
	app.run()
