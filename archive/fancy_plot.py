

import numpy as np

# from matplotlib import rcParams
# # change font to Arial
# # you can change this to any TrueType font that you have in your machine
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']

import matplotlib.pyplot as plt

# Generate two sets of numbers from a normal distribution
# one with mean = 4 sd = 0.5, another with mean (loc) = 1 and sd (scale) = 2
randomSet = np.random.normal(loc=4, scale=0.5, size=1000)
anotherRandom = np.random.normal(loc=1, scale=2, size=1000)
# Define a Figure and Axes object using plt.subplots
# Axes object is where we do the actual plotting (i.e. draw the histogram)
# Figure object is used to configure the actual figure (e.g. the dimensions of the figure)
fig, axs = plt.subplots(2)
# Plot a histogram with custom-defined bins, with a blue colour, transparency of 0.4
# Plot the density rather than the raw count using normed = True
axs[0].hist(randomSet, bins=np.arange(-3, 6, 0.5), color='#134a8e', alpha=0.4)
# Plot solid line for the means
axs[0].axvline(np.mean(randomSet), color='blue')
# Plot dotted lines for the std devs
axs[0].axvline(np.mean(randomSet) - np.std(randomSet), linestyle='--', color='blue')
axs[0].axvline(np.mean(randomSet) + np.std(randomSet), linestyle='--', color='blue')
# Set the title, x- and y-axis labels
axs[0].set_xlabel("Value of $x$")
axs[0].set_ylabel("Density")

axs[1].hist(anotherRandom, bins=np.arange(-3, 6, 0.5), color='#e8291c', alpha=0.4)
axs[1].axvline(np.mean(anotherRandom), color='red')
axs[1].axvline(np.mean(anotherRandom) - np.std(anotherRandom), linestyle='--', color='red')
axs[1].axvline(np.mean(anotherRandom) + np.std(anotherRandom), linestyle='--', color='red')
axs[1].set_xlabel("Value of $x$")
axs[1].set_ylabel("Density")

plt.suptitle("A fancy plot")
# Set the Figure's size as a 5in x 5in figure
fig.set_size_inches((5, 5))
