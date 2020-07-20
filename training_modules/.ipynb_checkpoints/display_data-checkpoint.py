{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'TYPE_NTRU' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-1-98f0b78db67f>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Load attacking traces\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;31m# TODO: fix this method up and make it work for ASCAD too.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0;32mdef\u001b[0m \u001b[0mdisplay_results\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maccuracies\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0madvantage\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mDB_TYPE\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mTYPE_NTRU\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mDB_TYPE\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32min\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mTYPE_ASCAD\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'TYPE_NTRU' is not defined"
     ]
    }
   ],
   "source": [
    "# Load attacking traces\n",
    "# TODO: fix this method up and make it work for ASCAD too.\n",
    "def display_results(models, accuracies, advantage, DB_TYPE = TYPE_NTRU):\n",
    "    \n",
    "    if DB_TYPE not in [TYPE_ASCAD]:\n",
    "    #     # we get 61% accuracy by just guessing, so let's compute the advantage over pure guesses:\n",
    "    #     print(\"+ recording calculating advantage\")\n",
    "    #     advantage = { i: (seed, key_idx, (accuracies[(seed, key_idx)] - .62) / (1-.62)) for i, (seed, key_idx) in enumerate(models.keys())}\n",
    "    #     # data = DataFrame.from_dict(advantage, orient='index', columns=['seed', 'key_idx', 'advantage'])\n",
    "    #     print(\"+ displaying data\")\n",
    "    #     # catplot(data=data, x=\"key_idx\", y=\"advantage\", row=\"seed\", kind=\"bar\")\n",
    "    #     print(\"advantage\", advantage)\n",
    "        adv = { i: (seed, key_idx, (advantage[(seed, key_idx)])) for i, (seed, key_idx) in enumerate(models.keys())}\n",
    "        print(\"adv: \", adv)\n",
    "        data = DataFrame.from_dict(adv, orient='index', columns=['seed', 'key_idx', 'adv'])\n",
    "        catplot(data=data, x=\"key_idx\", y=\"adv\", row=\"seed\", kind=\"bar\")\n",
    "\n",
    "        plt.savefig(get_file_path(training_model, ADV_GRPH_CONST))\n",
    "\n",
    "\n",
    "    # # # TODO: Needs to be moved to save_file(). You also need to call it before display results but after end of training\n",
    "    # # def save_adv(accuracy, save_location, seed, key_idx):\n",
    "    # #     advantage = (accuracies[(seed, key_idx)] - .62) / (1-.62)\n",
    "    # #     file_path = save_location + '/advantage' + '{:02d}'.format(seed)+\"key\"+'{:02d}'.format(key_idx)+\".txt\"\n",
    "    # #     np.savetxt(, advantage)\n",
    "    # #     return advantage\n",
    "    else:\n",
    "        adv = { i: (seed, (advantage[(seed)])) for i, (seed) in enumerate(models.keys())}\n",
    "        print(\"adv: \", adv)\n",
    "        data = DataFrame.from_dict(adv, orient='index', columns=['seed', 'adv'])\n",
    "        catplot(data=data, y=\"adv\", x=\"seed\", kind=\"bar\")\n",
    "        plt.savefig(get_file_path(training_model, ADV_GRPH_CONST))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sample_traces(unlab_traces):\n",
    "    \n",
    "    t = np.arange(unlab_traces.shape[1])\n",
    "    myTraces = np.array(unlab_traces)\n",
    "\n",
    "    rand_ind = np.zeros(tot)\n",
    "    for i in range(tot):\n",
    "        rand_ind[i] = np.random.randint(myTraces.shape[0])\n",
    "\n",
    "    # plot traces\n",
    "    print(\"++ plot traces\")\n",
    "    fig1, axs1 = plt.subplots(tot)\n",
    "    fig1.suptitle(DB_title + \": \" + str(rand_ind))\n",
    "    for i in range(tot):\n",
    "        axs1[i].plot(t, myTraces[int(rand_ind[i])])\n",
    "\n",
    "    # TODO: This frequency graphs were built by learning how the library was used in the following link:\n",
    "    # https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html    \n",
    "    # frequency plot\n",
    "    print(\"++ plot frequencies\")\n",
    "    fig2, axs2 = plt.subplots(tot)\n",
    "    fig2.suptitle(DB_title + \" Freqs: \" + str(rand_ind))\n",
    "    for j in range(tot):\n",
    "        Four, freqs = four_tr(myTraces, int(rand_ind[j]), f_s)\n",
    "    #     fig, ax = plt.subplots()\n",
    "        axs2[-j].stem(freqs, np.abs(Four), use_line_collection=True, linefmt='-', markerfmt=\" \")\n",
    "    #     axs2[j].set_xlim(-f_s / 2, f_s / 2)\n",
    "    #     axs2[j].set_ylim(-5, 110)\n",
    "\n",
    "    # zoomed in frequency plot\n",
    "    # print(\"++  plot frequencies (zoomed in)\")\n",
    "    # fig3, axs3 = plt.subplots(tot)\n",
    "    # fig3.suptitle(DB_title + \" Freqs: \" + str(rand_ind))\n",
    "    # for k in range(tot):\n",
    "    #     Four, freqs = four_tr(myTraces, int(rand_ind[k]), f_s)\n",
    "    #     Four[0] = 0\n",
    "    #     axs3[k].stem(freqs, np.abs(Four), use_line_collection=True, linefmt='-', markerfmt=\" \")\n",
    "    #     axs3[k].set_xlim(-10, 10)\n",
    "    #     axs3[k].set_ylim(-1, 10)\n",
    "\n",
    "    \n",
    "    \n",
    "    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html\n",
    "    print(\"++ plot spectrogram\")\n",
    "    fig4, axs4 = plt.subplots(tot)\n",
    "    fig4.suptitle(DB_title + \" sprectrogram: \" + str(rand_ind))\n",
    "    fig4.axes[-1].set_xlabel('time (sec)')\n",
    "    fig4.axes[int(tot/2)].set_ylabel('frequency')\n",
    "\n",
    "    for l in range(tot):\n",
    "        f, t, Sxx = signal.spectrogram(myTraces[int(rand_ind[l])], f_s)\n",
    "        axs4[l].pcolormesh(t, f, Sxx)\n",
    "#         axs4[l].set_xlim(-1, 20)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
