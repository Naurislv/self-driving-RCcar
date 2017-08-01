"""Used to listen and return keystroke inputs in a loop.

Run: main_loop()
Exit: ctrl+c
"""

import sys
import tty
import termios

class _Getch:

    def __call__(self):
        _fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(_fd)
        try:
            tty.setraw(sys.stdin.fileno())
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(_fd, termios.TCSADRAIN, old_settings)
        return char


def wait_key():
    """Main loop where computer is listening to keystroke inputs."""

    k = INKEY()
    return STROKES.get(k)

INKEY = _Getch()

STROKES = {}
STROKES['w'] = 'forward'
STROKES['s'] = 'backward'
STROKES['d'] = 'right'
STROKES['a'] = 'left'
STROKES['m'] = 'mode'
STROKES['n'] = 'next'
STROKES['o'] = 'stop'
STROKES['\x03'] = False

if __name__ == '__main__':
    RECV = True
    while RECV is not False:
        RECV = wait_key()
        print(RECV)
