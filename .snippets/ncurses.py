# Copyright(c) 2023 Rameez Ismail - All Rights Reserved
# Author: Rameez Ismail
# Email: rameez.ismaeel@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import curses
import time


def draw_progress_bar(window, percent):
    """
    Draws a progress bar on the given ncurses window.
    """
    max_y, max_x = window.getmaxyx()
    

    window.clear()
    window.addstr(0, 0, "Epoch 1")

    # Draw progress bar
    bar_width = max_x - 20  # Leave space for borders
    progress_width = int(percent * bar_width)
    window.addstr(0, 10, "[" + "#" * progress_width + " " * (bar_width - progress_width) + "]")

    # Add percentage
    percentage_str = "{:.0%}".format(percent)
    window.addstr(0, max_x-len(percentage_str), percentage_str)

    window.refresh()


def draw():
    # Initialize ncurses
    stdscr = curses.initscr()
    curses.noecho()
    #curses.cbreak()
    stdscr.keypad(True)

    try:
        # Draw initial progress bar
        window = curses.newwin(3, 40, 0, 0)
        draw_progress_bar(window, 0)

        # Update progress bar
        for i in range(101):
            draw_progress_bar(window, i/100)
            time.sleep(0.1)

    finally:
        # Clean up ncurses
        #curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()


if __name__ == "__main__":
   draw()