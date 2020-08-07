import numpy as np

from scipy import signal, fftpack
import matplotlib.pyplot as plt

from training_modules.misc import TYPE_ASCAD, TYPE_NTRU, TYPE_GAUSS, TYPE_DPA, TYPE_M4SC





        
def sample_traces(unlab_traces):
    
    t = np.arange(unlab_traces.shape[1])
    myTraces = np.array(unlab_traces)

    rand_ind = np.zeros(tot)
    for i in range(tot):
        rand_ind[i] = np.random.randint(myTraces.shape[0])

    # plot traces
    print("++ plot traces")
    fig1, axs1 = plt.subplots(tot)
    fig1.suptitle(DB_title + ": " + str(rand_ind))
    for i in range(tot):
        axs1[i].plot(t, myTraces[int(rand_ind[i])])

    # TODO: This frequency graphs were built by learning how the library was used in the following link:
    # https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html    
    # frequency plot
    print("++ plot frequencies")
    fig2, axs2 = plt.subplots(tot)
    fig2.suptitle(DB_title + " Freqs: " + str(rand_ind))
    for j in range(tot):
        Four, freqs = four_tr(myTraces, int(rand_ind[j]), f_s)
    #     fig, ax = plt.subplots()
        axs2[-j].stem(freqs, np.abs(Four), use_line_collection=True, linefmt='-', markerfmt=" ")
    #     axs2[j].set_xlim(-f_s / 2, f_s / 2)
    #     axs2[j].set_ylim(-5, 110)

    # zoomed in frequency plot
    # print("++  plot frequencies (zoomed in)")
    # fig3, axs3 = plt.subplots(tot)
    # fig3.suptitle(DB_title + " Freqs: " + str(rand_ind))
    # for k in range(tot):
    #     Four, freqs = four_tr(myTraces, int(rand_ind[k]), f_s)
    #     Four[0] = 0
    #     axs3[k].stem(freqs, np.abs(Four), use_line_collection=True, linefmt='-', markerfmt=" ")
    #     axs3[k].set_xlim(-10, 10)
    #     axs3[k].set_ylim(-1, 10)

    
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    print("++ plot spectrogram")
    fig4, axs4 = plt.subplots(tot)
    fig4.suptitle(DB_title + " sprectrogram: " + str(rand_ind))
    fig4.axes[-1].set_xlabel('time (sec)')
    fig4.axes[int(tot/2)].set_ylabel('frequency')

    for l in range(tot):
        f, t, Sxx = signal.spectrogram(myTraces[int(rand_ind[l])], f_s)
        axs4[l].pcolormesh(t, f, Sxx)
#         axs4[l].set_xlim(-1, 20)


