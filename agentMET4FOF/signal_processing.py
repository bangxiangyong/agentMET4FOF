import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

from agentMET4FOF.agents import AgentMET4FOF

matplotlib.use("Agg") # for sending plots as images

class FFT_Agent(AgentMET4FOF):
    """
    Buffers the received signal and apply fft on the collected signal upon reaching the `buffer_size` limit.
    The buffer then empties and the fft spectrum data and its plot are sent out on `default` and `plot` channel respectively.

    Parameters
    ----------
    s_freq : int
        Actual sampling frequency for plotting the x-axis
    """

    def init_parameters(self, s_freq=1):
        self.s_freq = s_freq

    def on_received_message(self, message):
        single_measurement = message["data"]
        agent_from = message["from"]
        self.buffer.store(data=single_measurement, agent_from=agent_from)

        if self.buffer.buffer_filled(agent_from):
            # extract collected signal from buffer
            buffered_sig = self.buffer[agent_from]

            # apply fft and convert to amplitudes
            sig_fft = fftpack.fft(buffered_sig)
            power_amp = np.abs(sig_fft) ** 2

            # get the corresponding frequencies
            sample_freq = fftpack.fftfreq(len(buffered_sig), d=1/self.s_freq)
            self.send_output(power_amp)
            self.send_plot(self.plot_fft(power_amp, sample_freq))
            self.buffer_clear(agent_from)

    def plot_fft(self, amplitudes, sample_freq):
        N = len(amplitudes)
        fig = plt.figure()
        plt.plot(sample_freq[1:N//2], amplitudes[1:N//2])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Amplitude")
        return fig
