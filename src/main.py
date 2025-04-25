from ui.gameui import GameUI
from game.game import Game
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Katruji / Zilatran Chess")
    game = Game.new_game()
    ui = GameUI(root, game)
    root.mainloop()
