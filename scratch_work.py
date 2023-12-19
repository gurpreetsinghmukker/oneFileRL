import matplotlib.pyplot as plt
import numpy as np
import random as random

x = np.random.rand(20)
print(x)

y = np.mean(x)
z = x[1:]
print(y)
print(z)

# # Enable interactive mode
# plt.ion()

# # Create a figure and axis
# fig, (ax1, ax2) = plt.subplots(2, 1)

# # Generate some initial data

# x1 = []
# x2 = []
# # print(len)
# y1 = [0]*len(x1)
# y2 = [0]*len(x2)
# # y1[0] = random.random()
# # y2[0] = random.random()

# # print(x)
# # print(y)

# # Create a line object which will be updated
# line1, = ax1.plot(x1, y1)
# line2, = ax2.plot(x2, y2)

# ymax1 = 0
# ymax2 = 0
# # Main loop
# max_steps = 100
# i = 0
# while i < max_steps:
    
#     # Update the data
#     y1.append(random.random())
#     x1.append(i)
#     if y1[i]>ymax1:
#         ymax1 = y1[i]
#         ax1.set_ylim(0, ymax1)


#     ax1.set_xlim(0,x1[i])
#     # Update the line object
#     line1.set_ydata(y1)
#     line1.set_xdata(x1)


#     if i%10 == 0:
#         y2.append(random.random())
#         if len(x2) == 0:
#             x2.append(0)
#         else:
#             x2.append(i/10)

#         if y2[int(i/10)]>ymax2:
#             ymax2 = y2[int(i/10)]
#             ax2.set_ylim(0, ymax2)
#         ax2.set_xlim(0,x2[int(i/10)])
#         line2.set_ydata(y2)
#         line2.set_xdata(x2)

#     # Redraw the figure
#     plt.draw()

#     # Pause execution for a short while to create animation effect
#     # plt.pause(0.1)
#     i += 1

# # Disable interactive mode
# plt.ioff()
# plt.show()