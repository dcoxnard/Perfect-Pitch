import flask
import pandas as pd
import numpy as np
import glob
import re
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
import librosa
import librosa.display
from io import StringIO
import keras

###############################################################
### PIPELINE ###
###############################################################
labels_filepath = './The Beatles Annotations/chordlab/The Beatles/*/*.lab'

labels = pd.DataFrame()
labels['label_file'] = glob.glob(labels_filepath)
labels['inferred_name'] = labels.label_file.apply(lambda x: re.sub(r'[0-9_\-]+', ' ', x).split('/')[-1].split('.lab')[0].split('CD')[-1].strip().lower())


features_filepath = './The Beatles/wav_files/*.wav'

features = pd.DataFrame()
features['feature_file'] = glob.glob(features_filepath)
features['inferred_name'] = features.feature_file.apply(lambda x: re.sub(r'[0-9]+', '', x).split('/')[-1].split('.wav')[0].strip().lower())


relative_map = {
    
    # Maj chords, all enharmonic equivalents
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'Ab':7,
    'Abb':7,
    'Bb':8,
    'Bbb':8,
    'Cb':1,
    'Db':9,
    'Eb':10,
    'Fb':4,
    'Gb':11,
    'A#':8,
    'B#':2,
    'C#':9,
    'D#':10,
    'E#':5,
    'F#':11,
    'G#':7,
    
    # Min chords, all enharmonic equivalents.
    'A min':2,
    'B min':3,
    'C min':10,
    'D min':5,
    'E min':6,
    'F min':7,
    'G min':8,
    'Ab min':1,
    'Bb min':9,
    'Cb min':3,
    'Db min':4,
    'Eb min':11,
    'Fb min':6,
    'Gb min':0,
    'A# min':9,
    'B# min':10,
    'C# min':4,
    'D# min':11,
    'E# min':7,
    'F# min':0,
    'G# min':1,
    
    # None
    'N': 12
}

###############################################################

def clean_ys(y):
    """
    Substitute complex chords into their simple maj/min versions.
    """

    for i, chord in enumerate(y):
        if not len(chord) == 1:
            chord = chord.split(':')
            tonality = re.sub(r'[0-9/]+', '', chord[0])
            flavor = chord[-1]
            if "min" in flavor:
                y[i] = tonality + ' min'
            else:
                y[i] = tonality
    return y

###############################################################

def get_timestamps_and_keys(song_name):
    """Return the list of timestamps of chord changes given a song name."""

    filepath = labels[labels.inferred_name.str.title() == song_name].label_file.values[0]

    timestamps = []
    keys = []

    with open(filepath, 'r') as f_obj:
            text = f_obj.readlines()
            inferred_name = re.sub(r'[0-9_\-]+', ' ', filepath).split('/')[-1].split('.lab')[0].split('CD')[-1].strip().lower()
            for line in text:
                line = line.split()        
                start = float(line[0])
                key = line[-1]
                timestamps.append(start)
                keys.append(key)
            # Grab the last timestamp in the song.
            timestamps.append(float(text[-1].split()[1]))

    keys = clean_ys(keys)
    keys = list(map(lambda x: relative_map[x], keys))

    return timestamps, keys

###############################################################

def generate_wavplot(song_name):
    """Save a .png of the wavfile and return the filepath that it's saved at"""

    filepath = features[features.inferred_name.str.title() == song_name].feature_file.values[0]
    rate, wave = wavfile.read(filepath)
    mono = np.mean(wave, axis=1)
    mono.shape
    plt.figure(figsize=(20,6))
    plt.axis('off')
    plt.plot(mono[::mono.shape[0]//6000], color='white')
    plt.tight_layout;
    friendly_song_name = '_'.join(song_name.split()).lower()
    output_filepath = './static/wavplots/' + friendly_song_name + '.png'
    plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0, transparent=True)
    return output_filepath

###############################################################

def trim(im):
    """Helper function for the colored_wavplot function."""
    
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

###############################################################

def colored_wavplot(song_name, timestamps, keys, downsample_rate=6000, matplotlib_cmap='hsv'):
    """Save a colored wavplot of the requested song, and return its saved filepath."""
    
    # Create numpy array from .wav file and downsample.
    filepath = features[features.inferred_name.str.title() == song_name].feature_file.values[0]
    rate, wave = wavfile.read(filepath)
    mono = np.mean(wave, axis=1)
    mono = mono[::mono.shape[0]//downsample_rate]  # Downsample

    # Generate colormap
    N = 13
    cmap = plt.cm.get_cmap(matplotlib_cmap, 13)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # Identify where color breaks happen.
    color_groups = []
    runtime = max(timestamps)
    for timestamp in timestamps:
        fraction = timestamp / runtime
        color_groups.append(int(fraction * mono.shape[0]))
        
    # Generate inputs into plt.plot() call.
    d = 0
    xs = []
    ys = []
    for color_group in color_groups[1:]:
        xs.append(np.arange(d, color_group))
        ys.append(mono[d:color_group])
        d = color_group
        
    # Create plot, trim, save, and return filepath.
    plt.figure(figsize=(20,2))
    plt.axis('off')
    for i in range(len(xs)):
        pitch_class = keys[i]
        plt.plot(xs[i], ys[i], color=cmaplist[pitch_class])
    plt.tight_layout;

    friendly_song_name = '_'.join(song_name.split()).lower()
    output_filepath = './static/wavplots/' + friendly_song_name + '.png'
    
    
    plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0, transparent=True)

    im = Image.open(output_filepath, 'r')
    im = trim(im)
    im.save(output_filepath)

    return output_filepath



###############################################################

def get_audio_file(song_name):

    filepath = features[features.inferred_name.str.title() == song_name].feature_file.values[0]
    return filepath


###############################################################
### MODEL ###
###############################################################

relative_map = {
    
    # Maj chords, all enharmonic equivalents
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'Ab':7,
    'Abb':7,
    'Bb':8,
    'Bbb':8,
    'Cb':1,
    'Db':9,
    'Eb':10,
    'Fb':4,
    'Gb':11,
    'A#':8,
    'B#':2,
    'C#':9,
    'D#':10,
    'E#':5,
    'F#':11,
    'G#':7,
    
    # Min chords, all enharmonic equivalents.
    'A min':2,
    'B min':3,
    'C min':10,
    'D min':5,
    'E min':6,
    'F min':7,
    'G min':8,
    'Ab min':1,
    'Bb min':9,
    'Cb min':3,
    'Db min':4,
    'Eb min':11,
    'Fb min':6,
    'Gb min':0,
    'A# min':9,
    'B# min':10,
    'C# min':4,
    'D# min':11,
    'E# min':7,
    'F# min':0,
    'G# min':1,
    
    # None
    'N': 12
}

###############################################################

def create_chromagram(start_sec, stop_sec, filename, channel='both'):
    
    rate, wave = wavfile.read(filename)
    start = int(start_sec * rate)
    stop = int(stop_sec * rate)
    
    # Can switch these guys in and out!
    left_channel = wave[start:stop,0] # One of the two stereo channels
    right_channel = wave[start:stop,1] # Second of the two stereo channels
    mean_channel = np.mean(wave[start:stop], axis=1) # Mean of the two stereo channels.  Might also be a way to get mono from scipy.
    
    if channel == 'both':
        wave = mean_channel.astype('float64')
    elif channel == 'left':
        wave = left_channel.astype('float64')
    elif channel == 'right':
        wave = right_channel.astype('float64')
    else:
        print("ERROR at {0}: PLEASE SELECT A VALID CHANNEL: {'left', 'right', 'both'}".format(filename))
    
    wave_harmonic, wave_percussive = librosa.effects.hpss(wave)
    
    C = librosa.feature.chroma_cqt(wave_harmonic, sr=rate)
    
    return C, rate




###############################################################

def add_padding(x, maxlen=500):
    """Pad/trim so that every sample has the same length."""
    
    # May want to increase maxlen from 500! Not sure the total dist of chomragram lengths.

    for i in range(len(x)):
        x[i] = x[i][:,:maxlen]
        q = maxlen - x[i].shape[1]
        p = q//2
#         if q % 2 == 0:
#             x[i] = np.pad(x[i], ((p,p), (0,0)), 'constant', constant_values=(0,0))
#         else:
#             x[i] = np.pad(x[i], ((p,p+1), (0,0)), 'constant', constant_values=(0,0))

        print
        if q % 2 == 0:
            x[i] = np.pad(x[i], ((0,0), (p,p)), 'constant', constant_values=(0,0))
        else:
            x[i] = np.pad(x[i], ((0,0), (p,p+1)), 'constant', constant_values=(0,0))
            
    return x




###############################################################

# Constructs X and Y.
# Took 37.5s for TEST_SIZE = 1.
# Took 5m 6s for TEST_SIZE = 10.
# Took 24m 56s for TEST_SIZE = 50.
# Took 2h 26m 4s for TEST_SIZE = 180.
# Took 1h 49m 53s for TEST_SIZE = 180 another time.

def construct_vars(lookup):
    """
    Pass the dictionary of songs/timestamps/keys.
    Create the corresponding x and y variables.
    """
    
    x_chroma = []
    y = []
    
    
    for k, v in lookup.items():
        
        inferred, start, stop = k
        musical_key = v

        filename = features[features.inferred_name == inferred].feature_file.values[0]

        try:
            x_chroma.append(create_chromagram(start, stop, filename, channel='both')[0])
            y.append(v)
        except:
            print("CHROMA ERROR:\nStart: {0}\nStop: {1}\nName: {2}\n\n\n".format(start, stop, inferred))
    
    x_chroma = add_padding(x_chroma)
    x = np.zeros((len(x_chroma), x_chroma[0].shape[0], x_chroma[0].shape[1]))
    for i in range(len(x)):
        x[i,:,:] = x_chroma[i]
    
    
    y = clean_ys(y)
    y = list(map(lambda x: relative_map[x], y))    # Switched out key_map  
    
    return x, y

###############################################################


def get_timestamps(filename, dictionary):
    """
    Insert name, start, stop and msucial key into the passed distionary.
    To use in iterating over annotation files.
    """
    
    with open(filename, 'r') as f_obj:
        text = f_obj.readlines()
        inferred_name = re.sub(r'[0-9_\-]+', ' ', filename).split('/')[-1].split('.lab')[0].split('CD')[-1].strip().lower()
        end_stamp = float(text[-1].split()[1])   # relic of an old idea.
        for line in text:
            line = line.split()        
            start = float(line[0])
            stop = float(line[1])
            musical_key = line[2]
            new_key = (inferred_name, start, stop)
            dictionary[new_key] = musical_key


###############################################################
model_savepath = 'dec10_test_model.h5'
model = keras.models.load_model(model_savepath)


###############################################################

# For demo.
# song_name = I Saw Her Standing There.

y_pred_hardcoded = [4, 4, 0, 4, 1, 4, 0, 0, 2, 4,
                    1, 4, 0, 4, 1, 4, 0, 0, 2, 4,
                    1, 4, 0, 1, 0, 4, 0, 4, 1, 4,
                    0, 2, 4, 1, 4, 4, 4, 0, 4, 0,
                    4, 0, 1, 0, 4, 0, 4, 1, 4, 0,
                    0, 2, 4, 1, 4, 1, 4, 1, 0, 0,
                    4, 12]

probs_hardcoded = [0.30241385,
                     0.90174961,
                     0.53617287,
                     0.91697454,
                     0.35139337,
                     0.4716121,
                     0.28480408,
                     0.79281527,
                     0.4415254,
                     0.70684123,
                     0.52354759,
                     0.94531286,
                     0.54144096,
                     0.95802259,
                     0.34432465,
                     0.53386366,
                     0.34266254,
                     0.77141178,
                     0.40955931,
                     0.71223009,
                     0.40662208,
                     0.80607009,
                     0.72866893,
                     0.39284682,
                     0.40612495,
                     0.81848979,
                     0.55085164,
                     0.90922451,
                     0.27508846,
                     0.82582748,
                     0.80808175,
                     0.50752753,
                     0.5903303,
                     0.37849727,
                     0.73843551,
                     0.29977426,
                     0.92609209,
                     0.63983876,
                     0.59841722,
                     0.27958024,
                     0.70726234,
                     0.61399806,
                     0.35262609,
                     0.38631004,
                     0.80617106,
                     0.51765603,
                     0.83834893,
                     0.25535318,
                     0.66514927,
                     0.23748206,
                     0.75137252,
                     0.46714652,
                     0.57431757,
                     0.43766975,
                     0.93448758,
                     0.36303568,
                     0.94663274,
                     0.32825744,
                     0.67180616,
                     0.28675279,
                     0.37876764,
                     0.63557887]


###############################################################
### APP ###
###############################################################

# Initialize the app.

app = flask.Flask(__name__)

@app.route('/')
def viz_page():
    with open("index.html", 'r') as viz_file:
        return viz_file.read()

@app.route('/list', methods=['POST'])
def get_songlist():
    songlist = list(labels.inferred_name.str.title())
    return flask.jsonify({"songlist": songlist})



@app.route('/song', methods=['POST'])
def process_song():

    # Get Data from request
    data = flask.request.json
    song_name = data['song_name']



    # Model.

    # lookup = {}

    # song_index = labels[labels.inferred_name.str.title() == song_name].index.values[0]

    # filename =  labels.iloc[song_index].label_file
    # get_timestamps(filename, lookup)

    # x, y = construct_vars(lookup)

    # x_reshaped = np.reshape(x, (x.shape[0], 12, 500, 1))

    # y_pred = [int(np.argmax(model.predict(np.reshape(x[i], (1, 12, 500, 1))))) for i in range(x.shape[0])]


    
    # Generate data.
    timestamps, keys = get_timestamps_and_keys(song_name)
    # wavplot_file = generate_wavplot(song_name)
    wavplot_file = colored_wavplot(song_name, timestamps, keys)
    audio_file = get_audio_file(song_name)

    # For debugging.
    print("SONG_NAME:", song_name)
    print("TIMESTAMPS:", timestamps, len(timestamps))
    print("Y_PRED:", y_pred_hardcoded)
    print("PROBS:", probs_hardcoded)
    print("KEYS:", keys, len(keys))
    print("WAVPLOT_FILE:", wavplot_file)
    print("AUDIO_FILE:", audio_file)

    return flask.jsonify({
        'timestamps': timestamps,
        'y_pred': y_pred_hardcoded,
        'keys': keys,
        'probs': probs_hardcoded,
        'wavplot_file': wavplot_file,
        'audio_file': audio_file,
        })

app.run(host='0.0.0.0', port=5000)