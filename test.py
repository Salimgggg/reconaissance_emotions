import matplotlib.pyplot as plt

# Example Y data
Y = [10, 8, 6, 4, 2]

# Create a new figure
fig = plt.figure()

# Add a subplot to the figure
ax = fig.add_subplot(1, 1, 1)

# Plot the Y data on the subplot
ax.plot(Y)

# Add labels to the X and Y axes
ax.set_xlabel("Index")
ax.set_ylabel("Y")

# Show the plot
plt.show()