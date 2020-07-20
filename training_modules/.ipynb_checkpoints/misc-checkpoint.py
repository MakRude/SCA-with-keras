{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Type constants defined as follows:\n",
    "class DB_TYPE:\n",
    "    TYPE_ASCAD = 0\n",
    "    TYPE_NTRU = 1\n",
    "    TYPE_GAUSS = 2\n",
    "    TYPE_DPA = 3\n",
    "    TYPE_M4SC = 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ASCAD: snippet from ASCAD code that's used throughout\n",
    "\n",
    "def check_file_exists(file_path):\n",
    "    file_path = os.path.normpath(file_path)\n",
    "    if os.path.exists(file_path) == False:\n",
    "        print(\"Error: provided file path '%s' does not exist!\" % file_path)\n",
    "        sys.exit(-1)\n",
    "    return\n"
   ]
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
