import numpy as np
import pandas as pd
from math import ceil
# import altair as alt
import librosa
# import librosa.display
from scipy.fft import rfft, rfftfreq
from scipy import signal
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, uic
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QStyle
from pyqtgraph import ViewBox, AxisItem, PlotDataItem
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import soundfile as sf
from datetime import datetime
from PyQt5.QtGui import QFont


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        self.actual_slider_names_dic = {}
        self.original_signal_output = None
        uic.loadUi(r'equalizer1.ui', self)

        self.setWindowTitle('Equalizer')
        # ui setup
        self.mode_name_comboBox.currentIndexChanged.connect(self.choose_mode)
        
        # browse
        self.Browse.clicked.connect(self.read_audio_file)

        # hide spectrogram
        self.hide_spectogram.stateChanged.connect(self.hide_the_spectograms)
        
        # to play and animate signal
        self.current_time_index = 0
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.update_signal_plot)
        self.play_pause = False
        self.last_pause_index = 0
        self.magnitude_time_after_inverse = None
        # state of the output
        self.output_state = False
        self.play_pause_button.clicked.connect(lambda: self.play(self.widget_input, self.magnitude_at_time, self.widget_output, self.magnitude_time_after_inverse))
        # set play icon
        self.play_pause_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        
        # speed
        self.speed_slider1.valueChanged.connect(self.update_speed)
        # Set the range of the speed slider
        self.speed_slider1.setRange(0, 400)
        # Set the single step, which is the increment or decrement value when using the slider
        self.speed_slider1.setSingleStep(5)
        # Set the initial value
        self.speed_slider1.setValue(10)
        
        # Connect buttons to zoom functions
        self.zoom_in.clicked.connect(lambda: self.zoom_graphs(0.5))
        self.zoom_out.clicked.connect(lambda: self.zoom_graphs(1.5))

        # reset
        self.reset.clicked.connect(self.reset_graphs)

        # spectrogram
        # Create a FigureCanvas instance
        self.canvas1 = FigureCanvas(plt.figure())
        self.canvas2 = FigureCanvas(plt.figure())
        # Create a QVBoxLayout instance
        self.layout1 = QVBoxLayout()
        self.layout2 = QVBoxLayout()
        # Add canvas to the layout
        self.layout1.addWidget(self.canvas1)
        self.layout2.addWidget(self.canvas2)

        # media player
        # Create QMediaPlayer instance
        self.media_player_before = QMediaPlayer()
        self.media_player_after = QMediaPlayer()

        # Connect media player to media source (audio file)
        self.setup_audio_player(self.play_button1, self.audio_slider_before, self.media_player_before)
        self.setup_audio_player(self.play_button2, self.audio_slider_after, self.media_player_after)
        
        # Connect the Media Player's positionChanged signal
        self.media_player_before.positionChanged.connect(self.media_player_position_changed)

        self.slider_names = [getattr(self, f'verticalSlider_{i + 1}') for i in range(10)]
        # set sliders range
        self.slider_range()
        
        # check if the slider value changed            
        for slider in self.slider_names:
                slider.valueChanged.connect(self.handle_slider_change)
        
        # check if the mode_name changed
        self.mode_name_comboBox.currentIndexChanged.connect(self.handle_mode_change)
        self.current_mode = self.mode_name_comboBox.itemText(0) #"Select A Mode..."
        
        self.uniform_slider_names_dic = {}
        
        # Hide them initially
        self.std_dev_label.hide()
        self.std_dev_line_edit.hide()
        self.std_dev_pushButton.hide()
        # Smoothing Window
        self.current_window_type = None
        self.used_window = {}
        self.rectangular_button.setChecked(True)
        # Connect the clicked signal of each window type radio button to the slot
        self.rectangular_button.clicked.connect(self.on_window_button_triggered)
        self.gaussian_button.clicked.connect(self.on_window_button_triggered)
        self.hann_button.clicked.connect(self.on_window_button_triggered)
        self.hamming_button.clicked.connect(self.on_window_button_triggered)

        # Connect the clicked signal of the apply button to the slot
        self.std_dev_pushButton.clicked.connect(self.on_apply_button_clicked)

    def handle_mode_change(self, index):
        # Stop the play timer
        self.play_timer.stop()
        self.play_pause = False

        # Reset the current time index
        self.current_time_index = 0
        self.magnitude_time_after_inverse = None
        self.original_signal_output = None
        
        # Clear all widgets and canvas
        self.clear_all_graphs()
        # If a second ViewBox exists, clear it
        if hasattr(self, 'vb2'):
            self.vb2.clear()
        self.canvas1.figure.clf()
        self.canvas2.figure.clf()
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
    
    def media_player_position_changed(self, position):
        # Calculate the corresponding index in the signal array based on the position
        index = int(position * self.sample_rate / 1000)  # Assuming position is in milliseconds

        # Update the signal plot
        self.update_audio_plot(index)
    
    def update_audio_plot(self, index):
        if index < len(self.time):
            x = self.time[:index]
            y_input = self.original_signal_input[:index]

            if self.original_signal_output is not None:
                y_output = self.original_signal_output[:index]
                self.widget_output.plot(x, y_output, pen='b', clear=True)
            self.widget_input.plot(x, y_input, pen='b', clear=True)
     
    def handle_slider_change(self):
        # modes dictionary
        modes_dict = {

            "Musical_Instruments": {
                self.verticalSlider_1: [0, 1000],     # drum
                self.verticalSlider_2: [1000, 2300],  # piano
                self.verticalSlider_3: [2000, 3000],  # xylophone
                self.verticalSlider_4: [3400, 25000]  # flute
            },
            "Animal": {
                self.verticalSlider_1: [650, 950],     # owl
                self.verticalSlider_2: [950, 1900],  # frog
                self.verticalSlider_3: [3000, 5500],  # canary
                self.verticalSlider_4: [6000, 19000]  # dolphin
            },
            "Medical": {
                self.verticalSlider_1: [25, 150],  # Arrythmia 1 (AF)
                self.verticalSlider_2: [7, 23],  # Arrythmia 2 (APC)
                self.verticalSlider_3: [1, 5],  # Arrythmia 3 (VT)
            }
        }
        # get current mode
        self.current_mode = self.mode_name_comboBox.currentText()
        if self.current_mode == "Select A Mode...":
            pass
        else:
            if self.current_mode == "Uniform_Range":
                # get slider names
                self.actual_slider_names = self.slider_names
            else:
                # get slider names and their ranges
                self.actual_slider_names_dic = modes_dict[self.current_mode]
                # get slider names
                self.actual_slider_names = list(self.actual_slider_names_dic.keys())
            # Get the slider that triggered the change
            slider = self.sender()
            std_dev = None #initialize
            if self.gaussian_button.isChecked() and self.std_dev_line_edit.text() != '':
                # Read the standard deviation from the line edit
                std_dev = float(self.std_dev_line_edit.text())
            # Call processing method
            self.processing(self.current_mode, self.magnitude_at_time, self.sample_rate, self.canvas2, self.layout2, slider, std_dev)
    
    def slider_range(self):
        for slider in self.slider_names:
                    slider.setMinimum(0)
                    slider.setMaximum(20)
                    slider.setSingleStep(1)
                    slider.setValue(10)

    def choose_mode(self):
        mode = self.mode_name_comboBox.currentText()

        if mode in ["Animal", "Musical_Instruments"]:
            category_lst = {
                "Animal": ["Owl", "Frog", "Canary", "Dolphin"],
                "Musical_Instruments": ["Drums", "Piano", "Xylophone", "Flute"]
            }
            lst = category_lst.get(mode, [])
            self.set_labels(lst, bold=True)
            self.set_labels(lst)
            self.hide_label()
            self.lab4.show()
            self.animal_and_music_mode()

        elif mode == "Uniform_Range":
            self.show_label()
            uniform_range_lst = ["0: 1000", "1000: 2000", "2000: 3000", "3000: 4000", "4000: 5000",
                                 "5000: 6000", "6000: 7000", "7000: 8000", "8000:9000", "9000:10000"]
            self.set_labels(uniform_range_lst, bold=True)
            self.Uniform_Range()

        elif mode == "Medical":
            medical_lst = ["Arrhythmia1", "Arrhythmia2", "Arrhythmia3"]
            self.set_labels(medical_lst, bold=True)
            self.hide_label()
            self.Medical()

        # set sliders range
        self.slider_range()
        
    def set_labels(self, lst, bold=False):
        for i, item in enumerate(lst):
            label = getattr(self, f'lab{i + 1}')
            label.setText(item)
            if bold:
                label.setFont(QFont("Arial", 10, QFont.Bold))

    def hide_not_required(self):
        for i in range(6):
            slider = getattr(self, f'verticalSlider_{i+5}')
            slider.hide()
            
    def hide_label(self):
        for i in range(7):
            label = getattr(self, f'lab{i + 4}')
            label.hide()

    def show_label(self):
        for i in range(7):
            label = getattr(self, f'lab{i + 4}')
            label.show()

    def animal_and_music_mode(self):
        self.hide_not_required()
        self.verticalSlider_4.show()
        self.show_required()
        self.play_button1.show()
        self.play_button2.show()

    def Uniform_Range(self):
        self.show_required()
        self.hide_audio()
        for i in range(10):
            slider = getattr(self, f'verticalSlider_{i+1}')
            slider.show()

    def show_required(self):
        self.groupBox_4.show()
        self.groupBox_3.show()
        self.groupBox_6.show()
        self.groupBox_7.show()
        self.audio_slider_after.show()
        self.audio_slider_before.show()
        self.label_7.show()
        self.label_8.show()
    
    def Medical(self):
        self.hide_not_required()
        self.show_required()
        self.verticalSlider_4.hide()
        self.hide_audio()
        
    def hide_audio(self):
        self.audio_slider_after.hide()
        self.audio_slider_before.hide()
        self.label_7.hide()
        self.label_8.hide()
        self.play_button1.hide()
        self.play_button2.hide()

    def hide_the_spectograms(self, state):
        # Check the state of the checkbox and hide/show the widget accordingly
        widgets_to_hide = [self.groupBox_4, self.groupBox_7]
        for widget in widgets_to_hide:
            if state == Qt.Checked:
                widget.hide()
            else:
                widget.show()
        return state

    def read_audio_file(self):
        self.audio_file, _ = QFileDialog.getOpenFileName(self, 'Open audio file', './', 'Audio Files (*.mp3 *.wav *.csv)')

        # if self.mode_name_comboBox.currentText() == "Medical":
        #     # Read the CSV file containing ECG data
        #     ecg_data = pd.read_csv(self.audio_file)
        #     self.time_X = ecg_data.iloc[:, 0]
        #     self.magnitude_at_time = ecg_data.iloc[:, 1]
        #     self.sample_rate = len(self.time_X)
        #     print("sr_B:", self.sample_rate)
        #     # self.play(self.widget_output,ecg_data.iloc[:,0],ecg_data.iloc[:,1])
        #     # self.plot_signal(self.widget_input, "Time", "s", "Amplitude", "mv", self.time_X, self.magnitude_at_time)

        self.media_player_before.setMedia(QMediaContent(QUrl.fromLocalFile(self.audio_file)))
        self.play_button1.setEnabled(True)

        self.magnitude_at_time, self.sample_rate = self.to_librosa(self.audio_file)
        self.time_X = np.linspace(0, self.magnitude_at_time.shape[0] / self.sample_rate,self.magnitude_at_time.shape[0])
            
        self.play(self.widget_input, self.magnitude_at_time, None, None)

        self.spectrogram(self.magnitude_at_time, self.mode_name_comboBox.currentText(), self.canvas1, self.layout1, self.widget_spect_in, "before")
        self.spectrum(self.mode_name_comboBox.currentText(), self.widget_spectrum, self.magnitude_at_time, self.sample_rate)
        self.pan(self.widget_input, self.time_X, self.magnitude_at_time)
        
    def clear_all_graphs(self):
        widget_lst = [self.widget_input, self.widget_output, self.widget_spectrum]
        for widget in widget_lst:
            widget.clear()
        
    def reset_sliders(self):
        for slider in self.actual_slider_names:
            slider.setValue(0)
        
    def play(self, widget_input, signal_input, widget_output=None, signal_output=None):
        if self.play_pause:
            self.play_timer.stop()
            self.last_pause_index = self.current_time_index
            self.play_pause = False
            self.play_pause_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        else:
            self.play_timer.setInterval(80)
            self.play_timer.start()
            self.play_pause = True
            self.play_pause_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
            self.current_widget_input = widget_input
            self.time = self.time_X
            self.original_signal_input = signal_input
            if widget_output and signal_output is not None:
                self.current_widget_output = widget_output
                self.original_signal_output = signal_output

    def setup_audio_player(self, play_button, audio_slider, media_player):
        play_button.setEnabled(False)
        play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        play_button.clicked.connect(lambda: self.play_audio(media_player))
        audio_slider.sliderMoved.connect(lambda position: self.set_audio_position(position, media_player))
        media_player.stateChanged.connect(lambda: self.media_state_changed(media_player, play_button))
        media_player.positionChanged.connect(lambda position: self.position_changed(position, audio_slider))
        media_player.durationChanged.connect(lambda d: self.duration_changed(d, audio_slider))

    def set_slider_properties(self, slider, min_value, max_value, step_size):
        slider.setRange(min_value, max_value)
        slider.setSingleStep(step_size)

    def plot_signal(self, widget, x_axis, x_axis_unit, y_axis, y_axis_unit,
                    x_data, y_data):
        # clear widget
        widget.clear()
        # plot
        widget.plot(x_data, y_data, pen='b')
        if self.mode_name_comboBox.currentText() == "Uniform_Range":
            widget.setXRange(0, 0.01)
        # Set labels and units
        widget.setLabel('bottom', text=x_axis, units=x_axis_unit)  # to set unit on x axis
        widget.setLabel('left', text=y_axis, units=y_axis_unit)
        # Add legend
        widget.addLegend()

    def update_signal_plot(self):
        if self.current_time_index < len(self.time):
            x = self.time[:self.current_time_index]
            y_input = self.original_signal_input[:self.current_time_index]

            if self.original_signal_output is not None:
                y_output = self.original_signal_output[:self.current_time_index]
                self.widget_output.plot(x, y_output, pen='b', clear=True)
            self.widget_input.plot(x, y_input, pen='b', clear=True)

            self.current_time_index += 50
        else:
            self.play_timer.stop()

    def update_speed(self, value):
        # link with audio
        playback_rate = value / 100.0

        # Set the new playback rate for both media players
        self.media_player_before.setPlaybackRate(playback_rate)
        self.media_player_after.setPlaybackRate(playback_rate)

        print(f"Slider value changed to: {value}")

        # Calculate the interval based on the slider value
        interval = 1000 / value if value > 0 else 1000

        # Adjust the timer interval for the selected channel
        self.play_timer.setInterval(interval)
        print("interval:", value)

    def zoom_graphs(self, factor):
        widgets_to_zoom = [self.widget_input, self.widget_output]
        for widget in widgets_to_zoom:
            widget.plotItem.getViewBox().scaleBy((factor, factor))
        # note factor will be 0.5 for zoom in ,1.5 for zoom out

    def reset_graphs(self):
        # Stop the play timer
        self.play_timer.stop()

        # Reset the current time index
        self.current_time_index = 0

        # Clear both input and output widgets
        self.widget_input.clear()
        self.widget_output.clear()
        self.play_pause = False

        # Play both input and output signals
        self.play(self.widget_input, self.magnitude_at_time, self.widget_output, self.magnitude_time_after_inverse)
        if self.output_state:
            self.play(self.widget_input, self.magnitude_at_time,self.widget_output, self.magnitude_time_after_inverse)

        # Reset the play/pause button icon
        self.play_pause_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))

        print("Graphs reset.")

        # elmafrood n3ml self.play(widget2,time of output, original signal of putput) bs lama n3ml el output
        # self.play(self.widget_output, self.time_of_output, self.original_signal_of_output)  keda a3taked

    def pan(self, widget, time, original_sig):
        widget.plotItem.setLimits(xMin=0, xMax=time[-1])
        widget.plotItem.setLimits(yMin=np.min(original_sig), yMax=np.max(original_sig))
        # widget.plotItem.setLimits(y=(np.min(original_sig), np.max(original_sig)))
    
    def to_librosa(self, file_uploaded):
        if file_uploaded is not None:
            y, sr = librosa.load(file_uploaded)
            return y, sr

    def play_audio(self, media_player):
        if media_player.state() == QMediaPlayer.PlayingState:
            media_player.pause()
        else:
            media_player.play()

    def set_audio_position(self, position, media_player):
        media_player.setPosition(position)

    def media_state_changed(self, media_player, play_button):
        if media_player.state() == QMediaPlayer.PlayingState:
            play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position, audio_slider):
        audio_slider.setValue(position)

    def duration_changed(self, d, audio_slider):
        audio_slider.setRange(0, d)

    def fourier_transform(self, audio_file, sample_rate):
        if not isinstance(audio_file, np.ndarray):
            # If audio_file is not a NumPy array, convert it
            audio_file = np.array(audio_file)
        number_samples = len(audio_file)
        T = 1 / sample_rate
        magnitude = rfft(audio_file)
        frequency = rfftfreq(number_samples, T)

        return magnitude, frequency

    def inverse_fourier_transform(self, magnitude_freq_domain):
        magnitude_time_domain = np.fft.irfft(magnitude_freq_domain)
        return np.real(magnitude_time_domain)

    def spectrogram(self, y, mode_name, canvas, layout, widget, title_of_graph):
        # if mode_name == "Medical" and self.canvas1:
        #     print("sr:", self.sample_rate)
        #     # Write the WAV file
        #     output_file_name = f"output1_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        #     sf.write(output_file_name, y, self.sample_rate)
        #     y, sr = self.to_librosa(output_file_name)
        # STFT of y
        D = librosa.stft(y)

        # apply logarithm to cast amplitude to Decibels
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        canvas.figure.clf()
        ax = canvas.figure.subplots()
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
        ax.set(title=title_of_graph)
        canvas.figure.colorbar(img, ax=ax, format="%+2.f dB")
        # self.canvas4.axes.plot(xData, self.reconstructed_signal)
        canvas.draw()
        widget.setLayout(layout)

    def spectrum(self, mode_name, widget, signal, sample_rate, slider_values = None):
        self.plot_frequency_spectrum(widget, signal, sample_rate)
        self.plot_window_functions(mode_name, widget, widget.plotItem, slider_values)
        
    def plot_frequency_spectrum(self, widget, signal, sample_rate):
        # Compute the Fourier transform
        magnitude_freq, frequency_freq = self.fourier_transform(signal, sample_rate)
        widget.clear()
        plot_item = widget.plotItem
        plot_item.setTitle("Spectrum")
        plot_item.setLabel('left', 'Magnitude')
        plot_item.setLabel('bottom', 'Frequency', units='Hz')
        # Plot the frequency spectrum
        widget.plot(frequency_freq, np.abs(magnitude_freq), pen='b')

    def plot_window_functions(self, mode_name, widget, plot_item, slider_values=None):
        # Plot the window functions for each slider on a second y-axis
        # if slider is not None:
        # Remove existing right axis
        if hasattr(self, 'ax2'):
            self.ax2.close()
            self.vb2.close()

        # Add new right axis
        self.ax2 = AxisItem('right')
        # Add it to the layout of the plot
        plot_item.layout.addItem(self.ax2, 2, 3)
        # Add new ViewBox and link it to the second y-axis
        self.vb2 = ViewBox()
        self.ax2.linkToView(self.vb2)
        if mode_name == 'Uniform_Range':
            slider_names_dic = self.uniform_slider_names_dic
        else:
            slider_names_dic = self.actual_slider_names_dic 

        # Initialize max_window_val to a very small number
        max_window_val = -float('inf')
        # Plot the window functions
        for slider, window in self.used_window.items():
            freq_range = slider_names_dic[slider]
            freq_array = np.linspace(freq_range[0], freq_range[1], len(window))
            slider_val = slider_values[slider]
            scaled_window = window * slider_val
            # Plot the scaled window
            window_plot = PlotDataItem(freq_array, scaled_window, pen='r')
            self.vb2.addItem(window_plot)
            # Update max_window_val
            max_window_val = max(max_window_val, max(scaled_window))

        # Add the second ViewBox to the scene of the plot
        plot_item.scene().addItem(self.vb2)
        # Make the second ViewBox share the x-axis with the original plot
        self.vb2.setGeometry(plot_item.vb.sceneBoundingRect())
        self.vb2.linkedViewChanged(plot_item.vb, self.vb2.XAxis)
        # Set the range of the y-axis for the window function plot
        self.vb2.setYRange(0, max_window_val) # max y-limit based on window plots' max value
        self.ax2.setLabel('Window Function')
        widget.show()

    def get_slider_values(self):
        slider_values = []
        slider_names = self.actual_slider_names
        for slider_name in slider_names:
            value = slider_name.value()
            # Check if the value is between 0 and 10
            if 0 <= value <= 10:
                # Map the value from 0 to 10 to the range 0 to 1
                normalized_value = value / 10.0
                slider_values.append(normalized_value)
            else:
                slider_values.append(value)
            print("slidervalue:",slider_values)

        # return slider_values_dic
        return dict(zip(slider_names, slider_values))

    def modifiy_signal(self, mode_name, magnitude_freq_domain, frequency_freq_domain, slider_values, slider, std_dev):        
        slider_names = self.actual_slider_names
        for i in range(len(slider_values)):
            if mode_name == "Uniform_Range":
                freq_range = self.uniform_slider_names_dic[slider_names[i]]
            else:
                freq_range = self.actual_slider_names_dic[slider_names[i]]
                
            freq_lower_lim = freq_range[0]
            freq_upper_lim = freq_range[1]
            freq_range_length = ceil(freq_upper_lim - freq_lower_lim)
            window = self.apply_smoothing_window(self.current_window_type, freq_range, std_dev)
            window_length = len(window)
            
            counter = 0
            for value in frequency_freq_domain:
                if value > freq_lower_lim and value < freq_upper_lim:
                   
                    # Calculate the index for the window array based on the ratio of the frequency range to the window length
                    window_index = int((value - freq_lower_lim) / freq_range_length * window_length)
                    # Ensure the index is within the bounds of the window array
                    window_index = min(window_index, window_length - 1)
                    
                    if slider_names[i] == slider:
                        magnitude_freq_domain[counter] *= window[window_index] * slider_values[slider_names[i]]
                        # Save the used window function for this slider
                        self.used_window[slider] = window

                    elif slider_names[i] in self.used_window:
                        magnitude_freq_domain[counter] *= window[window_index] * slider_values[slider_names[i]]
                        # Update the value in the dictionary
                        self.used_window[slider_names[i]] = window

                    else:
                        magnitude_freq_domain[counter] = magnitude_freq_domain[counter]
                
                counter += 1
        return magnitude_freq_domain

    def calculate_uniform_ranges(self, total_frequency, num_sliders):
        step_size = total_frequency // num_sliders
        ranges = [(i * step_size, (i + 1) * step_size) for i in range(num_sliders)]
        uniform_slider_names_dic = dict(zip(self.actual_slider_names, ranges))
        print("ranges:", ranges)
        return uniform_slider_names_dic

    def audio_after_modification(self, magnitude_time_after_inverse, sample_rate):
        # Generate a unique file name based on the current timestamp
        output_file_name = f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        # Write the audio data to the unique file name
        sf.write(output_file_name, magnitude_time_after_inverse, sample_rate)
        self.media_player_after.setMedia(QMediaContent(QUrl.fromLocalFile(output_file_name)))
        self.play_button2.setEnabled(True)

    # then apply FT -> then equalize -> then inverse FT to get back to time domain
    def processing(self, mode_name, magnitude_at_time, sample_rate, canvas, layout, slider=None, std_dev=None):
        magnitude_freq_domain, frequency_freq_domain = self.fourier_transform(magnitude_at_time, sample_rate)
        
        if mode_name == "Uniform_Range":
            total_frequency = max(frequency_freq_domain)
            self.uniform_slider_names_dic = self.calculate_uniform_ranges(total_frequency, len(self.actual_slider_names))
        slider_values = self.get_slider_values()
        
        # magnitude in frequency domain after modification
        magnitude_after_modify = self.modifiy_signal(mode_name, magnitude_freq_domain, frequency_freq_domain, slider_values, slider, std_dev)
        self.magnitude_time_after_inverse = self.inverse_fourier_transform(magnitude_after_modify)

        self.spectrum(mode_name, self.widget_spectrum, self.magnitude_time_after_inverse, sample_rate, slider_values)
        self.spectrogram(self.magnitude_time_after_inverse, mode_name, canvas, layout, self.widget_spect_out, "After")
        
        # display output signal in time domain
        self.play_pause = False
        # pan the output
        # self.pan(self.widget_output, self.time_X, self.magnitude_time_after_inverse)
        # display output signal in time domain
        self.play(self.widget_input, self.magnitude_at_time, self.widget_output, self.magnitude_time_after_inverse)
        if mode_name == "Musical_Instruments" or "Animal":
            self.audio_after_modification(self.magnitude_time_after_inverse, sample_rate)

    def on_apply_button_clicked(self):
        # Read the standard deviation from the line edit
        std_dev = float(self.std_dev_line_edit.text())
        # Apply the Gaussian window with the given standard deviation
        self.processing(self.current_mode, self.magnitude_at_time, self.sample_rate, self.canvas2, self.layout2, None, std_dev)

    def on_window_button_triggered(self):
        # Get the action that triggered the slot
        button = self.sender()
        # Check if the Gaussian window is selected
        if button == self.gaussian_button:
            # Show the label, line edit, and push button
            self.std_dev_label.show()
            self.std_dev_line_edit.show()
            self.std_dev_pushButton.show()
            self.current_window_type = 'Gaussian'
            std_dev = None #initialize
            if self.gaussian_button.isChecked() and self.std_dev_line_edit.text() != '':
                # Read the standard deviation from the line edit
                std_dev = float(self.std_dev_line_edit.text())
            # Call processing method
            self.processing(self.current_mode, self.magnitude_at_time, self.sample_rate, self.canvas2, self.layout2, std_dev)
        else:
            # Hide the label, line edit, and push button
            self.std_dev_label.hide()
            self.std_dev_line_edit.hide()
            self.std_dev_pushButton.hide()
            # Apply the other window types
            if button == self.hann_button:
                self.current_window_type = 'Hann'
            elif button == self.hamming_button:
                self.current_window_type = 'Hamming'
            elif button == self.rectangular_button:
                self.current_window_type = 'Rectangular'
            # Call processing method
            self.processing(self.current_mode, self.magnitude_at_time, self.sample_rate, self.canvas2, self.layout2)            
    
    def apply_smoothing_window(self, window_type=None, freq_range=None, std_dev=None):        
        # No smoothing window was chosen - Default
        if window_type is None:
            window_type = 'Rectangular'
        
        if freq_range is not None:            
            # Calculate the length of the window
            window_length = ceil(freq_range[1] - freq_range[0])

            # Create the smoothing window based on the window type
            if window_type == 'Hann':
                window = signal.windows.hann(window_length)
            elif window_type == 'Hamming':
                window = signal.windows.hamming(window_length)
            elif window_type == 'Rectangular':
                window = signal.windows.boxcar(window_length)
            elif window_type == 'Gaussian':
                if std_dev is None: # Default Standard Deviation
                    window = signal.windows.gaussian(window_length, std=window_length/6)
                else:
                    window = signal.windows.gaussian(window_length, std=std_dev)

            return window


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
