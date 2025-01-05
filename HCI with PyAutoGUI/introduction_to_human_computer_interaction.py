'''
Mouse Control Functions
Screen Size and Mouse Position:
Locations on your screen are referred to by X and Y Cartesian coordinates as demonstrated below:

0,0       X increases -->
+---------------------------+
|                           | Y increases
|                           |     |
|   1920 x 1080 screen      |     |
|                           |     V
|                           |
|                           |
+---------------------------+ 1919, 1079
The screen resolution size is returned by the size() function as a tuple of two integers. The current X and Y coordinates of the mouse cursor are returned by the position() function.

Function Syntax 
    size = pyautogui.size()
    location = pyautogui.position()
Mouse Movement
The moveTo() function will move the mouse cursor to the X and Y integer coordinates you pass it.

Function Syntax 
    pyautogui.moveTo(X, Y, duration)
X X coordinate of destination point.
Y Y coordinate of destination point.
duration optional flag of time (in seconds) the movement should take.
If you want to move the mouse cursor over a few pixels relative to its current position, use the move() function. This function has similar parameters as moveTo().

Mouse Drag
PyAutoGUI’s dragTo() and drag() functions have similar parameters as the moveTo() and move() functions. In addition, they have a button keyword which can be set to 'left', 'middle', and 'right' for which mouse button to hold down while dragging.

Function Syntax 
    pyautogui.dragTo( X, Y, duration, button )
X X coordinate of destination point.
Y Y coordinate of destination point.
duration optional flag of time (in seconds) the movement should take.
button button to press while dragging the mouse cursor.
Mouse Clicks
The click() function simulates a single, left-button mouse click at the mouse’s current position. A “click” is defined as pushing the button down and then releasing it up.

Function Syntax 
    pyautogui.click(X, Y, button, clicks, interval)
X optional X coordinate of destination point.
Y optional Y coordinate of destination point.
button button to press while dragging the mouse cursor.
clicks number of clicks to perform
interval specify the amount of pause between the clicks in seconds.
PyAutoGUI Documentation
Mouse control functions are just a small part of the PyAutoGUI library. For more information on the library, visit the PyAutoGUI documentation page.
'''




# Wait 2 seconds, to give you time to switch to the drawing application.
import time
import pyautogui
time.sleep(2.0)

distance = 200
while distance > 0:
        pyautogui.drag(distance, 0, button='left', duration=0.5)   # move right
        distance -= 50
        pyautogui.drag(0, distance, button='left', duration=0.5)   # move down
        
        pyautogui.drag(-distance, 0, button='left', duration=0.5)  # move left
        distance -= 50
        pyautogui.drag(0, -distance, button='left', duration=0.5)  # move up
        
        
'''
Keyboard Control Functions
The write() Function
The primary keyboard function is write(). This function will type the characters in the string that is passed.

Function Syntax 
    pyautogui.write(string, interval)
string string to type.
interval to add a delay interval in between pressing each character key.
The press(), keyDown(), and keyUp() Functions
To press these keys, call the press() function and pass it a string from the pyautogui.KEYBOARD_KEYS such as enter, esc, f1. See KEYBOARD_KEYS.

Function Syntax 
    pyautogui.press(key, presses, interval)
key string to denote which button to press.
presses number of key presses.
interval to add a delay interval in between pressing the key
The press() function is really just a wrapper for the keyDown() and keyUp() functions.

Hotkeys
To make pressing hotkeys or keyboard shortcuts convenient, the hotkey() can be passed several key strings which will be pressed down in order, and then released in reverse order.

Example:
    pyautogui.hotkey('ctrl', 'shift', 'esc')
To add a delay interval in between each press, pass an int or float for the interval keyword argument. Above example is same as:

    pyautogui.keyDown('ctrl')
    pyautogui.keyDown('shift')
    pyautogui.keyDown('esc')
    pyautogui.keyUp('esc')
    pyautogui.keyUp('shift')
    pyautogui.keyUp('ctrl')
PyAutoGUI Documentation
Keyboard control functions
'''


import time
import pyautogui

# Give a moment (half a second) to bring up the application window if needed.
time.sleep(0.5)

# If on a mac OSX machine, use command key instead of ctrl.
hotkey = 'command' if 'mac' in pyautogui.platform.platform() else 'ctrl'

# Open a new tab using a shortcut key.
pyautogui.hotkey(hotkey, 't')

# Give time for the browser to open the tab and be ready for user (typing) input.
time.sleep(1.0)

# Now type a url at a speedy 100 words per minute!
pyautogui.write('https://pyautogui.readthedocs.io', 0.01)

# Bring 'focus' to the URL bar (shortcut key may vary depending on your browser).
time.sleep(0.1)
pyautogui.hotkey(hotkey, 'l')

# Press enter to load the page.
pyautogui.press('enter')